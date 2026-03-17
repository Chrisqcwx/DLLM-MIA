def _generate_calibration_texts(self, text: str, n_cal: int = 8) -> list[str]:
    """
    生成轻微扰动版本作为 null 分布估计。
    这些变体在语义上相似但不太可能在训练集中。
    """
    tokens = self.tokenizer.encode(text)
    L = len(tokens)
    calibration_texts = []
    
    # 策略 1：随机滑动窗口截取不同片段
    for _ in range(n_cal // 2):
        start = np.random.randint(0, max(1, L // 4))
        end = min(L, start + int(L * 0.9))
        sub_tokens = tokens[start:end]
        calibration_texts.append(self.tokenizer.decode(sub_tokens))
    
    # 策略 2：随机 shuffle 句子顺序（打乱顺序，内容不变但不再是原文）
    sentences = text.split('. ')
    if len(sentences) > 2:
        for _ in range(n_cal // 2):
            shuffled = sentences.copy()
            np.random.shuffle(shuffled)
            calibration_texts.append('. '.join(shuffled))
    
    return calibration_texts[:n_cal]

def _self_calibrated_score(self, text: str, raw_score: float, n_cal: int = 8) -> float:
    """
    用扰动变体的分数分布来校准 raw_score。
    """
    cal_texts = self._generate_calibration_texts(text, n_cal)
    if not cal_texts:
        return raw_score
    
    # 计算扰动变体的分数（复用已有 batch score 逻辑）
    cal_scores = self._compute_batch_scores(cal_texts, batch_start_idx=-1)
    
    mu = np.mean(cal_scores)
    sigma = np.std(cal_scores) + 1e-8
    return float((raw_score - mu) / sigma)  # z-score
