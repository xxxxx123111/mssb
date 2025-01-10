import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# **1. 加载训练数据**
def load_training_data(positive_file, negative_file):
    def extract_reviews(file_path, label):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 提取 <review> 标签内的文本
        reviews = re.findall(r'<review id="\d+">\s*(.*?)\s*</review>', content, re.S)
        return pd.DataFrame({'text': reviews, 'label': label})
    
    positive_df = extract_reviews(positive_file, 1)  # 正向评论
    negative_df = extract_reviews(negative_file, 0)  # 负向评论
    return pd.concat([positive_df, negative_df], ignore_index=True)

# 加载中文和英文训练数据
cn_data = load_training_data('sample.positive.cn.txt', 'sample.negative.cn.txt')
en_data = load_training_data('sample.positive.en.txt', 'sample.negative.en.txt')

# **2. 数据预处理**
def preprocess_data(data, language="CN"):
    if language == "CN":  # 中文分词
        data['text'] = data['text'].apply(lambda x: ' '.join(jieba.cut(x)))
    else:  # 英文小写化
        data['text'] = data['text'].apply(lambda x: x.lower())
    return data

# 预处理数据
cn_data = preprocess_data(cn_data, language="CN")
en_data = preprocess_data(en_data, language="EN")

# **3. 文本向量化**
def vectorize_text(data, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(data['text'])
    return pad_sequences(sequences, maxlen=max_len)

# 中文向量化
tokenizer_cn = Tokenizer(num_words=10000)
tokenizer_cn.fit_on_texts(cn_data['text'])
X_cn = vectorize_text(cn_data, tokenizer_cn)
y_cn = cn_data['label']

# 英文向量化
tokenizer_en = Tokenizer(num_words=10000)
tokenizer_en.fit_on_texts(en_data['text'])
X_en = vectorize_text(en_data, tokenizer_en)
y_en = en_data['label']

# 划分训练集和验证集
X_cn_train, X_cn_val, y_cn_train, y_cn_val = train_test_split(X_cn, y_cn, test_size=0.2, random_state=42)
X_en_train, X_en_val, y_en_train, y_en_val = train_test_split(X_en, y_en, test_size=0.2, random_state=42)

# **4. 模型构建与训练**
def build_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=100),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 中文模型训练
model_cn = build_model()
model_cn.fit(X_cn_train, y_cn_train, batch_size=32, epochs=10, validation_data=(X_cn_val, y_cn_val))
model_cn.save('sentiment_model_cn.h5')

# 英文模型训练
model_en = build_model()
model_en.fit(X_en_train, y_en_train, batch_size=32, epochs=10, validation_data=(X_en_val, y_en_val))
model_en.save('sentiment_model_en.h5')

# **5. 测试数据预测**
def load_test_data(file_path, language="CN"):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    reviews = re.findall(r'<review id="\d+">\s*(.*?)\s*</review>', content, re.S)
    data = pd.DataFrame({'text': reviews})
    return preprocess_data(data, language)

# 加载测试数据
test_cn = load_test_data('test.cn.txt', language="CN")
test_en = load_test_data('test.en.txt', language="EN")

# 中文测试数据预测
X_test_cn = vectorize_text(test_cn, tokenizer_cn)
model_cn = load_model('sentiment_model_cn.h5')
predictions_cn = model_cn.predict(X_test_cn)
test_cn['predicted_label'] = (predictions_cn > 0.5).astype(int)

# 英文测试数据预测
X_test_en = vectorize_text(test_en, tokenizer_en)
model_en = load_model('sentiment_model_en.h5')
predictions_en = model_en.predict(X_test_en)
test_en['predicted_label'] = (predictions_en > 0.5).astype(int)

# **6. 格式化输出**
def format_output(data, lang):
    output = []
    for i, row in data.iterrows():
        label = "positive" if row['predicted_label'] == 1 else "negative"
        output.append(f"TeamName 1 {i} {label}")
    return output

# 中文结果格式化
formatted_cn = format_output(test_cn, lang='zh')

# 英文结果格式化
formatted_en = format_output(test_en, lang='en')

# **7. 保存结果**
with open('TeamName_1_CN.txt', 'w', encoding='utf-8') as f:
    for line in formatted_cn:
        f.write(line + '\n')

with open('TeamName_1_EN.txt', 'w', encoding='utf-8') as f:
    for line in formatted_en:
        f.write(line + '\n')

print("中文预测结果已保存至 formatted_predictions_cn.txt")
print("英文预测结果已保存至 formatted_predictions_en.txt")
