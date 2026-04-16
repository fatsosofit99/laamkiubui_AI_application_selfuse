
import math
from collections import defaultdict, Counter
from transformers import AutoTokenizer

class SimpleTokenizer:
    def __init__(self, model_name="./bert-base-uncased"):
        # 注意：如果本地没有该模型，会自动从 HuggingFace 下载
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> list:
        return self.tokenizer.tokenize(text.lower())

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.word_log_probs = {}
        self.vocab = set()
        self.tokenizer = SimpleTokenizer()
        self.class_word_counts = {}
        self.class_total_words = {}
        self.classes = []

    def _extract_features(self, text: str) -> Counter:
        tokens = self.tokenizer.tokenize(text)
        return Counter(tokens)
    
    def _print_top_words_per_class(self, top_k=5):
        print("条件概率 Top 词（每类前 %d 个）:" % top_k)
        for c in self.word_log_probs:
            sorted_words = sorted(self.word_log_probs[c].items(), key=lambda x: x[1], reverse=True)
            print(f"  类别 {c}:")
            for word, log_prob in sorted_words[:top_k]:
                prob = math.exp(log_prob)
                print(f"    P({word}|{c})={prob:.4f}")

    def compute_prior_probs(self, labels: list) -> dict:
        """
        目标 1：计算对数先验概率 log P(y)
        """
        total_samples = len(labels)
        label_counts = Counter(labels)
        priors = {}
        
        for label, count in label_counts.items():
            # 公式：log P(y) = log(类别样本数 / 总样本数)
            priors[label] = math.log(count / total_samples)
            
        return priors
    #TODO
    def compute_likelihood_probs(self, features: list, labels: list) -> tuple[dict, dict, set]:
        """
        目标 2：计算对数条件概率（带拉普拉斯平滑）
        """
        word_counts_per_class = defaultdict(Counter)
        class_total_words = defaultdict(int)
        vocab = set()

        # 统计频词表、类总词数和全局词汇表
        for feature, label in zip(features, labels):
            for word, count in feature.items():
                word_counts_per_class[label][word] += count
                class_total_words[label] += count
                vocab.add(word)
        #TODO
        vocab_size = len(vocab)
        word_log_probs = defaultdict(dict)

        # 计算每个词在每个类别下的对数条件概率
        for label in set(labels):
            denominator = class_total_words[label] + vocab_size
            for word in vocab:
                # 如果该词在当前类中未出现，则为 0
                word_count = word_counts_per_class[label].get(word, 0)
                # 拉普拉斯平滑公式
                prob = (word_count + 1) / denominator
                word_log_probs[label][word] = math.log(prob)

        return dict(word_log_probs), dict(class_total_words), vocab
        #TODO
    def predict(self, text):
        """
        目标 3：计算给定文本在各个类别下的对数后验概率，返回得分字典
        """
        tokens = self.tokenizer.tokenize(text)
        class_scores = {}
        vocab_size = len(self.vocab)

        for label in self.classes:
            # 初始化为先验概率
            score = self.class_priors[label]
            denominator = self.class_total_words[label] + vocab_size
            
            for word in tokens:
                if word in self.vocab:
                    # 若词在词表中，加上预先计算好的对数条件概率
                    score += self.word_log_probs[label][word]
                else:
                    # 对于 OOV (未登录词)，使用平滑概率并取对数
                    oov_prob = 1.0 / denominator
                    score += math.log(oov_prob)
            
            class_scores[label] = score
            
        return class_scores
        #TODO
    def fit(self, texts: list, labels: list) -> None:
        self.classes = list(set(labels))
        self.class_priors = self.compute_prior_probs(labels)
        print("log 先验概率:", self.class_priors)

        features = [self._extract_features(text) for text in texts]
        self.word_log_probs, self.class_total_words, self.vocab = self.compute_likelihood_probs(features, labels)

        print("每类总词数:", dict(self.class_total_words))
        print("词表大小:", len(self.vocab))
        
        self._print_top_words_per_class(top_k=5)


if __name__ == "__main__":
    texts = [
        "I love this movie, it's fantastic and thrilling!",
        "Terrible film. Waste of time.",
        "Absolutely amazing story and great acting.",
        "Worst movie I've ever seen.",
        "It was okay, not the best but not the worst."
    ]
    labels = ['pos', 'neg', 'pos', 'neg', 'neutral']

    model = NaiveBayesClassifier()
    model.fit(texts, labels)

    print("\n--- 测试 ---")
    test_text = "What a great and thrilling story!"
    class_scores = model.predict(test_text)
    print("预测文本:", test_text)
    print("预测结果:", class_scores)
    print("预测标签:", max(class_scores, key=class_scores.get))