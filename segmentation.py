import cv2
import numpy as np


class _KMeans:
    """简易的K-Means实现，用于GMM初始化"""

    def __init__(self, n_clusters, n_features, max_iters=10, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
        self.labels = None

    def _initialize_centroids(self, X):
        # 从数据点中随机选择初始质心
        random_indices = np.random.choice(
            X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices].astype(np.float64)

    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters), dtype=np.float64)
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:  # 如果一个簇没有点，重新随机初始化它（或者保持不变）
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def fit(self, X):
        if X.shape[0] == 0:  # 没有数据点
            self.labels = np.array([], dtype=int)
            # 确保质心不会是 uninitialized if K-Means is expected to run
            self.centroids = np.zeros(
                (self.n_clusters, self.n_features), dtype=np.float64)
            return self.labels

        if X.shape[0] < self.n_clusters:
            # 如果数据点少于簇数
            self.centroids = np.zeros(
                (self.n_clusters, self.n_features), dtype=np.float64)
            num_points_to_use = X.shape[0]
            self.centroids[:num_points_to_use] = X[:num_points_to_use].astype(
                np.float64)
            if num_points_to_use < self.n_clusters and num_points_to_use > 0:
                for i in range(num_points_to_use, self.n_clusters):
                    self.centroids[i] = X[0].astype(np.float64)  # 用第一个点填充不足的质心
            elif num_points_to_use == 0:
                pass
            self.labels = self._assign_clusters(X)
            return self.labels

        self._initialize_centroids(X)
        for iteration in range(self.max_iters):
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)
            if np.sum(np.linalg.norm(self.centroids - old_centroids, axis=1)) < self.tol:
                break
        return self.labels


class _CustomGMM:
    """自定义高斯混合模型 (GMM)"""

    def __init__(self, n_components=5, n_features=3, reg_covar=1e-6):
        self.n_components = n_components
        self.n_features = n_features
        self.reg_covar = reg_covar

        # GMM 参数
        self.weights = np.ones(n_components, dtype=np.float64) / n_components
        self.means = np.zeros((n_components, n_features), dtype=np.float64)
        self.covariances = np.array(
            [np.eye(n_features, dtype=np.float64) for _ in range(n_components)]) * 0.1

        self._inv_covariances = np.zeros_like(self.covariances)
        self._det_covariances = np.zeros(n_components, dtype=np.float64)
        self.initialized = False
        self._update_cov_determinants_and_inverses()

    def _multivariate_gaussian_pdf(self, X, mean, inv_cov, det_cov):
        """计算多元高斯分布的概率密度函数值"""
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        if det_cov <= 0:
            return np.zeros(X.shape[0])

        log_sqrt_2pi_det_cov = 0.5 * \
            (self.n_features * np.log(2 * np.pi) + np.log(det_cov))
        log_pdf = -log_sqrt_2pi_det_cov + exponent

        pdf_vals = np.exp(log_pdf)
        return pdf_vals

    def _update_cov_determinants_and_inverses(self):
        for k in range(self.n_components):
            cov_k = self.covariances[k]

            reg_matrix = np.eye(self.n_features) * self.reg_covar
            cov_k_reg = cov_k + reg_matrix

            try:

                self._inv_covariances[k] = np.linalg.inv(cov_k_reg)
                det_val = np.linalg.det(cov_k_reg)

                if det_val <= 1e-40:
                    stronger_reg = self.reg_covar + 1e-3
                    cov_k_reg_stronger = cov_k + \
                        np.eye(self.n_features) * stronger_reg

                    try:
                        self._inv_covariances[k] = np.linalg.inv(
                            cov_k_reg_stronger)
                        det_val = np.linalg.det(cov_k_reg_stronger)
                        if det_val <= 1e-40:
                            self.covariances[k] = np.eye(
                                self.n_features) * 0.1
                            self._inv_covariances[k] = np.eye(
                                self.n_features) / 0.1
                            det_val = np.linalg.det(self.covariances[k])
                    except np.linalg.LinAlgError:

                        self.covariances[k] = np.eye(self.n_features) * 0.1
                        self._inv_covariances[k] = np.eye(
                            self.n_features) / 0.1
                        det_val = np.linalg.det(self.covariances[k])

                self._det_covariances[k] = max(det_val, 1e-40)

            except np.linalg.LinAlgError:

                self.covariances[k] = np.eye(self.n_features) * 0.1
                self._inv_covariances[k] = np.eye(self.n_features) / 0.1
                self._det_covariances[k] = np.linalg.det(self.covariances[k])

    def fit(self, X, component_assignments):
        """
        根据给定的数据点X和它们对应的GMM分量指派来更新GMM参数 (M-step)
        X: (n_samples, n_features)
        component_assignments: (n_samples,) 每个样本分配到的GMM分量索引
        """
        if X.shape[0] == 0:

            if not self.initialized:
                self.weights = np.ones(
                    self.n_components, dtype=np.float64) / self.n_components
                self.means = np.zeros(
                    (self.n_components, self.n_features), dtype=np.float64)
                self.covariances = np.array(
                    [np.eye(self.n_features, dtype=np.float64) for _ in range(self.n_components)]) * 0.1
                self._update_cov_determinants_and_inverses()
                self.initialized = True
            return

        n_samples = X.shape[0]
        responsibilities_sum = np.zeros(self.n_components, dtype=np.float64)

        for k in range(self.n_components):
            X_k = X[component_assignments == k]
            n_k = X_k.shape[0]
            responsibilities_sum[k] = n_k

            if n_k == 0:  # 这个分量没有分配到任何点
                self.weights[k] = 1e-9  # 避免权重为0, 保持旧的均值和协方差
                continue

            self.weights[k] = n_k / n_samples
            self.means[k] = np.mean(X_k, axis=0)

            if n_k <= self.n_features:
                self.covariances[k] = np.eye(self.n_features) * 0.1
            else:
                diff = X_k - self.means[k]
                self.covariances[k] = (diff.T @ diff) / n_k

        current_sum_weights = np.sum(self.weights)
        if current_sum_weights > 1e-8:
            self.weights /= current_sum_weights
        else:
            self.weights = np.ones(
                self.n_components, dtype=np.float64) / self.n_components

        self._update_cov_determinants_and_inverses()
        self.initialized = True

    def predict_component_proba(self, X):
        """计算X中每个样本属于每个GMM分量的概率 P(x | component_k)"""
        if not self.initialized:
            return np.ones((X.shape[0], self.n_components)) / self.n_components

        proba = np.zeros((X.shape[0], self.n_components), dtype=np.float64)
        for k in range(self.n_components):

            if self.weights[k] < 1e-8 or self._det_covariances[k] <= 0:
                proba[:, k] = 1e-12
                continue
            proba[:, k] = self._multivariate_gaussian_pdf(
                X, self.means[k], self._inv_covariances[k], self._det_covariances[k])
        return proba

    def assign_components(self, X):
        """为X中的每个样本分配最可能的GMM分量 (E-step的一部分)"""
        if not self.initialized:
            if X.shape[0] > 0:
                return np.random.randint(0, self.n_components, X.shape[0])
            return np.array([], dtype=int)

        component_probas = self.predict_component_proba(X)
        weighted_probas = component_probas * self.weights
        assignments = np.argmax(weighted_probas, axis=1)
        return assignments

    def calculate_total_likelihood(self, X):
        """计算X中每个样本的GMM总似然度 P(x|GMM) = sum_k P(x|component_k) * P(k)"""
        if not self.initialized:
            return np.zeros(X.shape[0])

        component_probas = self.predict_component_proba(X)
        weighted_probas = component_probas * self.weights
        total_likelihood = np.sum(weighted_probas, axis=1)
        return total_likelihood


def _initialize_mask_and_trimap(image_shape, mask_input, roi_tuple, mode_option,
                                scribbles_fg_points, scribbles_bg_points, brush_size):
    """
    初始化alpha图 (trimap)。
    返回: grabcut_mask (包含0,1,2,3的mask), 和 rect_for_grabcut_api (可能为None)
    此函数合并了原`run_grabcut_logic`中对mask的初步处理和`_apply_scribbles_to_mask_internal`。
    """
    height, width = image_shape[:2]
    grabcut_mask = np.zeros((height, width), dtype=np.uint8)

    rect_for_grabcut_api = roi_tuple

    if mode_option == "INIT_WITH_RECT":

        grabcut_mask[:, :] = cv2.GC_BGD
        if roi_tuple and roi_tuple[2] > 0 and roi_tuple[3] > 0:
            x, y, w, h = roi_tuple
            grabcut_mask[y:y+h, x:x+w] = cv2.GC_PR_FGD

        grabcut_mask = _apply_scribbles_to_mask_internal(
            grabcut_mask, scribbles_fg_points, scribbles_bg_points, brush_size)

    elif mode_option == "INIT_WITH_MASK":
        grabcut_mask = mask_input.copy()
        grabcut_mask = _apply_scribbles_to_mask_internal(
            grabcut_mask, scribbles_fg_points, scribbles_bg_points, brush_size)
        rect_for_grabcut_api = None

    elif mode_option == "EVAL":
        grabcut_mask = mask_input.copy()
        grabcut_mask = _apply_scribbles_to_mask_internal(
            grabcut_mask, scribbles_fg_points, scribbles_bg_points, brush_size)
        rect_for_grabcut_api = None
    else:
        raise ValueError("无效的GrabCut初始化模式选项。")

    return grabcut_mask


def _run_custom_grabcut_internal(image, initial_alpha_mask, iterations, n_gmm_components=5):
    """
    自定义GrabCut的核心逻辑。
    """
    height, width = image.shape[:2]
    pixels_rgb = image.reshape(-1, 3).astype(np.float64)
    alpha_flat = initial_alpha_mask.reshape(-1).copy()

    fg_gmm = _CustomGMM(n_components=n_gmm_components, n_features=3)
    bg_gmm = _CustomGMM(n_components=n_gmm_components, n_features=3)

    # 前景 GMM 初始化: 优先使用 GC_FGD, 若无则使用 GC_PR_FGD
    fg_init_indices = np.where(alpha_flat == cv2.GC_FGD)[0]
    if len(fg_init_indices) == 0:
        fg_init_indices = np.where(alpha_flat == cv2.GC_PR_FGD)[0]

    if len(fg_init_indices) > 0:
        fg_init_pixels = pixels_rgb[fg_init_indices]
        if len(fg_init_pixels) >= n_gmm_components:
            kmeans_fg = _KMeans(n_clusters=n_gmm_components, n_features=3)
            fg_component_assignments = kmeans_fg.fit(fg_init_pixels)
            fg_gmm.fit(fg_init_pixels, fg_component_assignments)
        else:  # 样本不足以进行KMeans，简化初始化
            fg_gmm.means[0] = np.mean(fg_init_pixels, axis=0)
            fg_gmm.covariances[0] = np.cov(fg_init_pixels.T) if len(
                fg_init_pixels) > 1 else np.eye(3) * 0.1
            if n_gmm_components > 1:  # 禁用其他分量
                for k_idx in range(1, n_gmm_components):
                    fg_gmm.weights[k_idx] = 1e-9
                fg_gmm.weights[0] = 1.0 - np.sum(fg_gmm.weights[1:])
            fg_gmm._update_cov_determinants_and_inverses()
            fg_gmm.initialized = True
    else:  # 没有可用于初始化前景GMM的像素 (GC_FGD 或 GC_PR_FGD 均为空)
        fg_gmm.initialized = True

    # 背景 GMM 初始化: 优先使用 GC_BGD, 若无则使用 GC_PR_BGD
    bg_init_indices = np.where(alpha_flat == cv2.GC_BGD)[0]
    if len(bg_init_indices) == 0:
        bg_init_indices = np.where(alpha_flat == cv2.GC_PR_BGD)[0]

    if len(bg_init_indices) > 0:
        bg_init_pixels = pixels_rgb[bg_init_indices]
        if len(bg_init_pixels) >= n_gmm_components:
            kmeans_bg = _KMeans(n_clusters=n_gmm_components, n_features=3)
            bg_component_assignments = kmeans_bg.fit(bg_init_pixels)
            bg_gmm.fit(bg_init_pixels, bg_component_assignments)
        else:  # 样本不足
            bg_gmm.means[0] = np.mean(bg_init_pixels, axis=0)
            bg_gmm.covariances[0] = np.cov(bg_init_pixels.T) if len(
                bg_init_pixels) > 1 else np.eye(3) * 0.1
            if n_gmm_components > 1:
                for k_idx in range(1, n_gmm_components):
                    bg_gmm.weights[k_idx] = 1e-9
                bg_gmm.weights[0] = 1.0 - np.sum(bg_gmm.weights[1:])
            bg_gmm._update_cov_determinants_and_inverses()
            bg_gmm.initialized = True
    else:
        bg_gmm.initialized = True

    # --- 迭代优化 ---
    unknown_indices = np.where(
        (alpha_flat == cv2.GC_PR_FGD) | (alpha_flat == cv2.GC_PR_BGD))[0]

    for i in range(iterations):
        # --- 步骤1: (可选的) 为硬标签像素(GC_FGD, GC_BGD)分配GMM内部分量 ---
        # 这一步主要是如果GMM有多个分量，确定哪个分量最适合这些硬标签像素
        # 对于GMM学习，步骤2更重要，它使用所有当前 FG/BG 估计。
        # known_fg_hard_indices = np.where(alpha_flat == cv2.GC_FGD)[0]
        # known_bg_hard_indices = np.where(alpha_flat == cv2.GC_BGD)[0]
        # if fg_gmm.initialized and len(known_fg_hard_indices) > 0:
        #     fg_gmm.assign_components(pixels_rgb[known_fg_hard_indices]) # assignments not directly used here
        # if bg_gmm.initialized and len(known_bg_hard_indices) > 0:
        #     bg_gmm.assign_components(pixels_rgb[known_bg_hard_indices])

        # --- 步骤2: 学习GMM参数 ---
        # 使用所有当前被认为是前景(FGD, PR_FGD)和背景(BGD, PR_BGD)的像素来更新GMM
        current_fg_indices = np.where(
            (alpha_flat == cv2.GC_FGD) | (alpha_flat == cv2.GC_PR_FGD))[0]
        current_bg_indices = np.where(
            (alpha_flat == cv2.GC_BGD) | (alpha_flat == cv2.GC_PR_BGD))[0]

        if fg_gmm.initialized and len(current_fg_indices) > 0:
            all_fg_pixels = pixels_rgb[current_fg_indices]
            all_fg_component_assignments = fg_gmm.assign_components(
                all_fg_pixels)
            fg_gmm.fit(all_fg_pixels, all_fg_component_assignments)
        # else: fg_gmm 可能未从数据初始化或当前没有前景像素

        if bg_gmm.initialized and len(current_bg_indices) > 0:
            all_bg_pixels = pixels_rgb[current_bg_indices]
            all_bg_component_assignments = bg_gmm.assign_components(
                all_bg_pixels)
            bg_gmm.fit(all_bg_pixels, all_bg_component_assignments)
        # else: bg_gmm 可能未从数据初始化或当前没有背景像素

        # --- 步骤3: 为未知区域像素估计分割 ---
        if len(unknown_indices) > 0 and fg_gmm.initialized and bg_gmm.initialized:
            unknown_pixels_data = pixels_rgb[unknown_indices]

            fg_likelihoods = fg_gmm.calculate_total_likelihood(
                unknown_pixels_data)
            bg_likelihoods = bg_gmm.calculate_total_likelihood(
                unknown_pixels_data)

            # 简单分类 (无平滑项)
            assign_to_fg = fg_likelihoods > bg_likelihoods

            # 更新 alpha_flat 中 unknown_indices 部分的标签
            # 注意: 保持 GC_FGD 和 GC_BGD (硬标签) 不变
            alpha_flat[unknown_indices[assign_to_fg]] = cv2.GC_PR_FGD
            alpha_flat[unknown_indices[~assign_to_fg]] = cv2.GC_PR_BGD
        elif len(unknown_indices) > 0:
            # print("警告: GMMs 未完全初始化，无法对未知区域进行分类。")
            pass

    return alpha_flat.reshape(height, width)


def _apply_scribbles_to_mask_internal(mask_to_modify, scribbles_fg_points, scribbles_bg_points, brush_size):
    """
    将涂鸦（(x,y)元组的列表）应用于GrabCut掩码。
    前景用 cv2.GC_FGD 标记。
    背景用 cv2.GC_BGD 标记。
    """
    # 前景涂鸦
    for scribble_path in scribbles_fg_points:
        if len(scribble_path) == 1:  # 单点
            p = scribble_path[0]
            cv2.circle(mask_to_modify, p, brush_size // 2, cv2.GC_FGD, -1)
        else:
            for i in range(len(scribble_path) - 1):
                p1 = scribble_path[i]
                p2 = scribble_path[i+1]
                cv2.line(mask_to_modify, p1, p2, cv2.GC_FGD, brush_size)

    # 背景涂鸦
    for scribble_path in scribbles_bg_points:
        if len(scribble_path) == 1:  # 单点
            p = scribble_path[0]
            cv2.circle(mask_to_modify, p, brush_size // 2, cv2.GC_BGD, -1)
        else:
            for i in range(len(scribble_path) - 1):
                p1 = scribble_path[i]
                p2 = scribble_path[i+1]
                cv2.line(mask_to_modify, p1, p2, cv2.GC_BGD, brush_size)
    return mask_to_modify


def run_grabcut_logic(original_image, mask_input, roi_tuple,
                      scribbles_fg_points, scribbles_bg_points, brush_size,
                      mode_option):
    """
    使用自定义实现执行GrabCut分割。
    """
    if original_image is None:
        return None, None, "错误：GrabCut缺少原始图像。"

    iterations = 5
    status_message = ""

    try:
        initial_grabcut_mask = _initialize_mask_and_trimap(
            original_image.shape, mask_input, roi_tuple, mode_option,
            scribbles_fg_points, scribbles_bg_points, brush_size
        )
    except ValueError as e:
        return mask_input, None, f"错误：{e}"

    if mode_option == "INIT_WITH_RECT":
        if not roi_tuple or roi_tuple[2] <= 0 or roi_tuple[3] <= 0:
            return mask_input, None, "错误：ROI对于INIT_WITH_RECT无效。"
        status_message = "自定义GrabCut已通过ROI初始化"
        if scribbles_fg_points or scribbles_bg_points:
            status_message += "并应用了涂鸦"
    elif mode_option == "INIT_WITH_MASK":
        status_message = "自定义GrabCut已通过掩码（和涂鸦）初始化/优化"
    elif mode_option == "EVAL":
        iterations = 1
        status_message = "自定义GrabCut正在评估优化"

    try:
        updated_grabcut_mask = _run_custom_grabcut_internal(
            original_image, initial_grabcut_mask, iterations
        )
    except Exception as e:
        # import traceback
        # traceback.print_exc() # For debugging errors in custom GrabCut
        return mask_input, None, f"自定义GrabCut期间出错：{e}"

    output_mask_binary = np.where((updated_grabcut_mask == cv2.GC_BGD) | (
        updated_grabcut_mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')

    return updated_grabcut_mask, output_mask_binary, status_message + "。自定义GrabCut完成。"


def apply_border_matting_logic(original_image, grabcut_result_mask):
    """
    应用边界抠图（使用引导滤波器或高斯模糊进行简化）。
    """
    if original_image is None or grabcut_result_mask is None:
        return None, "错误：抠图缺少图像或GrabCut掩码。"

    try:
        binary_mask_fg = np.where((grabcut_result_mask == cv2.GC_FGD) | (grabcut_result_mask == cv2.GC_PR_FGD),
                                  255, 0).astype(np.uint8)
        src_for_filter = binary_mask_fg.astype(np.float32) / 255.0
        guide_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        alpha_matte = None
        matting_status = "已应用边界抠图"

        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            alpha_matte = cv2.ximgproc.guidedFilter(guide=guide_image_gray, src=src_for_filter,
                                                    radius=10, eps=0.01**2)  # eps should be small, like 0.01^2 or 0.001^2
            alpha_matte = np.clip(alpha_matte, 0, 1)
            matting_status += " (引导滤波器)。"
        else:
            alpha_matte = cv2.GaussianBlur(src_for_filter, (15, 15), 0)
            matting_status += " (高斯模糊备选方案 - 请安装 opencv-contrib-python 以使用引导滤波器)。"

        alpha_matte_uint8 = (alpha_matte * 255).astype(np.uint8)
        b, g, r = cv2.split(original_image)
        matted_image_bgra = cv2.merge((b, g, r, alpha_matte_uint8))
        return matted_image_bgra, matting_status

    except Exception as e:
        return None, f"边界抠图期间出错：{e}"
