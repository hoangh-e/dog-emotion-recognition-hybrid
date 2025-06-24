Dưới đây là **bảng chi tiết mở rộng** các phương pháp ensemble theo đúng cấu trúc yêu cầu, bao gồm bốn trường: **Chiến lược đa lớp** (multiclass strategy), **Hạn chế chính**, **Hiệu năng**, **Sử dụng (Python/Package)** cùng với **Cách áp dụng & lý do phù hợp** và **Nguồn nghiên cứu**.

---

### 1. **Bagging (Bootstrap Aggregating)**

* **Chiến lược đa lớp**: Homogeneous ensemble (đa lớp qua nhiều mô hình giống nhau)
* **Hạn chế chính**: Giảm phương sai nhưng nếu base learner có bias cao thì cũng không cải thiện; cần huấn luyện nhiều mô hình – tốn thời gian và tài nguyên ([xgboosting.com][1], [xgboosting.com][2])
* **Hiệu năng**: \~8/10 (tốt nếu base là high-variance, như ResNet); cải thiện ổn định đáng kể
* **Sử dụng**: Keras/TensorFlow để train nhiều ResNet/YOLO; vote cuối bằng majority (numpy/scikit) hoặc sử dụng `sklearn.ensemble.BaggingClassifier`
* **Cách áp dụng & lý do phù hợp**: Bootstrapping dữ liệu giúp giảm phương sai; phù hợp khi dữ liệu lớn, model deep dễ overfit
* **Nguồn nghiên cứu**: Ganaie et al. (2021/2022) review Bagging trong deep ensembles ([arxiv.org][3]); Wikipedia giải thích rõ bootstrap sampling ([en.wikipedia.org][4])

---

### 2. **Boosting**

* **Chiến lược đa lớp**: Sequential ensemble
* **Hạn chế chính**: Dễ overfit nếu không regularize tốt; chậm hơn Bagging do tuần tự; không parallel hóa tốt ([colab.research.google.com][5], [machinelearningmastery.com][6])
* **Hiệu năng**: 9/10 (cực tốt khi dùng XGBoost/LightGBM trên tập mẫu hiếm)
* **Sử dụng**: `xgboost.XGBClassifier(objective="multi:softprob"/"softmax")` hoặc `lightgbm.LGBMClassifier()`
* **Cách áp dụng & lý do phù hợp**: Tập trung học sample khó, đặc biệt khi lớp unbalanced (ví dụ: cảm xúc hiếm)
* **Nguồn nghiên cứu**: Ganaie et al. tổng quát boosting ([sciencedirect.com][7], [colab.research.google.com][5], [arxiv.org][3]); Springer's review ensemble methods ([link.springer.com][8])

---

### 3. **Stacking (Stacked Generalization)**

* **Chiến lược đa lớp**: Heterogeneous base + meta-model (đa mô hình base + model meta)
* **Hạn chế chính**: Phức tạp khi tạo data cho meta (need OOF preds), dễ overfit nếu tập nhỏ ([analyticsvidhya.com][9])
* **Hiệu năng**: 9/10 (tốt nhất khi base models đa dạng, errors ít liên quan)
* **Sử dụng**: `sklearn.ensemble.StackingClassifier(estimators=[...], final_estimator=XGBClassifier() or LogisticRegression())`
* **Cách áp dụng & lý do phù hợp**: Kết hợp ResNet/YOLO + classifiers truyền thống để tận dụng strengths của từng model
* **Nguồn nghiên cứu**: Ganaie et al. categorizations ([xgboosting.com][1]); MachineLearningMastery hướng dẫn stacking deep nets ([machinelearningmastery.com][10])

---

### 4. **Soft Voting / Hard Voting**
* **Chiến lược đa lớp**: Simple ensemble
* **Hạn chế chính**: Không tối ưu khi models có độ tin cậy rất khác nhau; soft voting dễ bị model yếu chi phối nếu không cân bằng ([stackoverflow.com][11])
* **Hiệu năng**: 7.5/10 (đơn giản, hiệu quả, dễ triển khai)
* **Sử dụng**: `sklearn.ensemble.VotingClassifier(estimators=[...], voting='soft'/'hard')`
* **Cách áp dụng & lý do phù hợp**: Dự đoán cảm xúc bằng cách cộng xác suất (soft) hoặc majority vote (hard) của ResNet, YOLO,...
* **Nguồn nghiên cứu**: Wikipedia ensemble&#x20;

---

### 5. **Negative Correlation Ensemble**

* **Chiến lược đa lớp**: Penalize correlation giữa learners
* **Hạn chế chính**: Cần custom training loop để thêm penalty; phức tạp setup training&#x20;
* **Hiệu năng**: 9/10 (tăng đa dạng, giảm correlation, nâng cao hiệu suất)
* **Sử dụng**: Deep NCL implementations – custom loss trong Keras/TensorFlow/PyTorch
* **Cách áp dụng & lý do phù hợp**: Huấn luyện ResNet/YOLO clones cùng backbone với loss gồm accuracy + correlation penalty
* **Nguồn nghiên cứu**: Deep Negative Correlation Classification 2022 ([arxiv.org][12]); GNCL 2020&#x20;

---

### 6. **Heterogeneous Ensemble**

* **Chiến lược đa lớp**: Vision + tail model combine
* **Hạn chế chính**: Harder để tune khi models khác biệt; cần đồng bộ xác suất đầu ra&#x20;
* **Hiệu năng**: 8.5/10 (phù hợp với multimodal features, tăng robustness)
* **Sử dụng**: `VotingClassifier` hoặc `StackingClassifier` với base gồm ResNet, YOLO, tail bằng XGBoost
* **Cách áp dụng & lý do phù hợp**: Kết hợp deep vision + classical tabular model để tận dụng cả probabilistic features và khả năng generalize
* **Nguồn nghiên cứu**: Ganaie et al. categorization ([xgboosting.com][1], [arxiv.org][3]); heterogeneous boosting meta-learner study ([link.springer.com][8])

---

### 7. **Multi-level Deep Ensembles**

* **Chiến lược đa lớp**: Implicit ensemble via feature fusion + meta learner
* **Hạn chế chính**: Tốn bộ nhớ lưu trữ và cần sync feature scales; tuning khó&#x20;
* **Hiệu năng**: 9.5/10 (rất cao nếu features đa cấp thực sự chứa complementary information)
* **Sử dụng**: Trích feature từ nhiều cấp deep nets (face/body/context) → `LightGBM` hoặc `XGBClassifier` dùng như meta learner
* **Cách áp dụng & lý do phù hợp**: Implicit ensemble giúp bỏ qua cần vote, dễ mở rộng theo cấp độ feature
* **Nguồn nghiên cứu**: Ganaie et al. discuss explicit/implicit ensembles ([arxiv.org][3]); Deep Sub-Ensembles concept&#x20;

---


[1]: https://xgboosting.com/stacking-ensemble-with-xgboost-meta-model-final-model/?utm_source=chatgpt.com "Stacking Ensemble With XGBoost Meta Model (Final Model)"
[2]: https://xgboosting.com/stacking-ensemble-with-one-xgboost-base-model-heterogeneous-ensemble/?utm_source=chatgpt.com "Stacking Ensemble With One XGBoost Base Model (Heterogeneous Ensemble ..."
[3]: https://arxiv.org/abs/2104.02395?utm_source=chatgpt.com "Ensemble deep learning: A review"
[4]: https://en.wikipedia.org/wiki/Ensemble_learning?utm_source=chatgpt.com "Ensemble learning"
[5]: https://colab.research.google.com/github/kaggler-tv/blog/blob/master/_notebooks/2021-04-26-stacking-ensemble.ipynb?utm_source=chatgpt.com "Stacking Ensemble - Google Colab"
[6]: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/?utm_source=chatgpt.com "Stacking Ensemble Machine Learning With Python"
[7]: https://www.sciencedirect.com/science/article/pii/S095219762200269X?utm_source=chatgpt.com "Ensemble deep learning: A review - ScienceDirect"
[8]: https://link.springer.com/chapter/10.1007/978-981-15-7345-3_60?utm_source=chatgpt.com "Evaluating Heterogeneous Ensembles with Boosting Meta-Learner - Springer"
[9]: https://www.analyticsvidhya.com/blog/2021/08/ensemble-stacking-for-machine-learning-and-deep-learning/?utm_source=chatgpt.com "Ensemble Stacking for Machine Learning and Deep Learning - Analytics Vidhya"
[10]: https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/?utm_source=chatgpt.com "Stacking Ensemble for Deep Learning Neural Networks in Python"
[11]: https://stackoverflow.com/questions/75812594/making-an-ensemble-learning-function-based-on-xgboost-models?utm_source=chatgpt.com "Making an ensemble learning function based on XGboost models"
[12]: https://arxiv.org/abs/2212.07070?utm_source=chatgpt.com "Deep Negative Correlation Classification"