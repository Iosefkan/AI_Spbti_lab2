# Отчет по второй лабораторной работе

## 1. Теоретическая база
**Цель работы:** научиться работать с предобученными моделями и на основе предобученных эмбеддингов строить новые модели.

---
### **Трансформеры**

**Архитектура трансформеров** была представлена в статье **"Attention is All You Need"** в 2017 году. Она является основой для современных моделей обработки естественного языка (NLP), таких как BERT, GPT, T5 и других.

### **Ключевые особенности**
- **Отказ от рекуррентных и свёрточных слоёв**:
- Трансформеры не используют RNN или CNN, что устраняет проблему последовательной обработки данных.
- **Механизм внимания (Self-Attention):**
- Позволяет учитывать взаимосвязи между всеми токенами в последовательности.
- **Параллелизация:**
- Обработка токенов осуществляется одновременно, что ускоряет обучение.

### **Компоненты архитектуры**

1. **Входные данные**
   - **Токенизация:** Текст преобразуется в числовые представления токенов.
   - **Эмбеддинги:** Каждому токену сопоставляется вектор фиксированного размера.
   - **Позиционные эмбеддинги:** Добавляются для учёта порядка токенов в последовательности.

2. **Механизм внимания (Attention Mechanism)**
   - Для каждого токена вычисляется важность (внимание) относительно всех остальных токенов.
   - Формула внимания:
        - Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

        где:
        - Q — запросы (*queries*),
        - K — ключи (*keys*),
        - V — значения (*values*),
        - d_k — размерность ключей.
3. **Мультиголовое внимание (Multi-Head Attention)**
   - Разделяет механизм внимания на несколько независимых "голов", каждая из которых фокусируется на разных аспектах взаимосвязей.
   - Объединённые результаты проходят через линейный слой.

4. **Обучаемые слои**
   - **Feed-Forward Network (FFN):** Полносвязная сеть, применяемая к каждому токену отдельно.
   - **Нормализация:** Layer Normalization стабилизирует обучение.
   - **Dropout:** Предотвращает переобучение.

5. **Энкодер и Декодер**
   - **Энкодер:**
     - Состоит из \(N\) одинаковых слоёв.
     - Каждый слой включает:
       - Мультиголовое внимание.
       - FFN.
     - Используется для преобразования входных данных в скрытые представления.
   - **Декодер:**
     - Состоит из \(N\) одинаковых слоёв.
     - Каждый слой включает:
       - Self-Attention.
       - Cross-Attention (внимание на выход энкодера).
       - FFN.
     - Используется для генерации выходной последовательности.

### Алгоритм работы трансформера

1. **Входные данные:** Последовательность токенов преобразуется в эмбеддинги с добавлением позиционной информации.
2. **Энкодер:**
   - Каждый токен проходит через несколько слоёв, включая self-attention и FFN.
   - На выходе формируется скрытое представление.
3. **Декодер:**
   - На основе выходов энкодера и собственных входов генерируется последовательность.
4. **Предсказание:** Декодер выдаёт итоговый результат (например, перевод текста).

---
### **BERT**

**BERT (Bidirectional Encoder Representations from Transformers)** — это модель обработки естественного языка (NLP), представленная исследовательской командой Google AI в 2018 году. Она основана на архитектуре трансформера и предназначена для обучения контекстных представлений текста.

**Основные особенности BERT**
- **Двунаправленность (Bidirectionality):**

    В отличие от моделей, обрабатывающих текст слева направо (например, GPT), BERT анализирует текст в обоих направлениях одновременно.
    Это позволяет модели учитывать контекст слова как из левой, так и из правой части предложения.
- **Предобучение на больших корпусах:**

    BERT обучается на огромных наборах данных, таких как Wikipedia и BookCorpus, с использованием двух основных задач:
    Masked Language Model (MLM): Некоторые токены маскируются (заменяются на [MASK]), и модель обучается предсказывать их.
    Next Sentence Prediction (NSP): Модель обучается предсказывать, является ли второе предложение продолжением первого.
- **Основа на трансформере:**

    BERT использует архитектуру Transformer Encoder, которая включает:
    Многоголовое внимание (Multi-Head Attention).
    Механизмы нормализации и пропуска (Residual Connections).

---
### **ALBERT**

**ALBERT (A Lite BERT)** — это улучшенная и более эффективная версия модели BERT, представленная Google Research в 2019 году. ALBERT сохраняет архитектуру трансформера, но включает оптимизации, которые снижают вычислительные затраты и потребление памяти.

**Основные улучшения ALBERT**
- **Факторизация матрицы эмбеддингов (Embedding Factorization):**

    - В BERT размер эмбеддингов слов (vocabulary embeddings) и скрытого слоя одинаков (например, 768 или 1024).
    - В ALBERT эти два размера разделены: эмбеддинги слов имеют меньшую размерность (например, 128), а затем проецируются в скрытое пространство через линейный слой.
    - Это уменьшает количество параметров, особенно для больших словарей.
- **Параметрическое деление (Parameter Sharing):**

    - В BERT каждый слой энкодера имеет свои собственные параметры.
    - В ALBERT все слои делят одни и те же параметры, что значительно сокращает количество параметров модели.
    - Деление происходит для:
        - Матриц весов в слоях внимания.
        - Матриц весов в промежуточных и выходных слоях.
- **Пересмотр задачи Next Sentence Prediction (NSP):**

    - ALBERT заменяет NSP на более сложную задачу Sentence Order Prediction (SOP), которая помогает лучше моделировать интер-связь предложений:
    - Модели подаются пары предложений, где:
        - Порядок нормальный (A, B).
        - Порядок нарушен (B, A).
    - Задача модели — определить правильный порядок.
- **Меньшее число параметров с сохранением производительности:**

    - За счёт факторизации эмбеддингов и деления параметров ALBERT содержит меньше параметров, чем BERT, но демонстрирует сопоставимую или лучшую производительность.
    
## 2. Алгоритм работы разработанной системы

1. **Подготовка данных:**
    - Изначальный текст приводится к нижнему регистру
    - Удаляются пунктуация и числа
    - Удаляются стоп-слова, как основа используетс список стоп-слов, загружаемый через библиотеку nltk
    - Удаляются эмодзи
    - Производится стемминг слов
2. **Подготовка данных:**
    - Очищенный CSV файл считывается в DataFrame
    - Выполняется валидация необходимых колонок
    - Маппинг категорий: текстовые категории преобразуются в числовые метки, создавая словарь соответствий.
3. **Конфигурация предобученной модели ALBERT:**
    - Используется предварительно обученная модель ALBERT:
    - AlbertTokenizer для токенизации текстов.
    - AlbertForSequenceClassification для классификации.
    - Параметры:
        - model_name = "albert-base-v2" — модель ALBERT базового размера.
        - max_length = 128 — максимальная длина токенов.
        - batch_size = 16 — размер мини-батча для предсказания.
4. **Токенизация:**
    - Применяются операции паддинга, усечения и преобразования в тензоры PyTorch.
    - Токенизированные данные отправляются на устройство (GPU или CPU).
4. **Обучение:**
    - Модель AlbertForSequenceClassification принимает токенизированные тексты и возвращает логиты — вероятности принадлежности к каждому классу.
    - Для каждой группы токенов из мини-батча вычисляется индекс класса с максимальной вероятностью.
    - Предсказания аккумулируются в список.
5. **Вычисление метрик:**
    - Метрики оцениваются на основе реальных и предсказанных меток:
    - Precision: доля корректно предсказанных положительных примеров.
    - Recall: доля корректно обнаруженных положительных примеров среди всех реальных.
    - F1 Score: гармоническое среднее между precision и recall.


## 3. Результат выполнения программы
```
Category Mapping: {'travel': 0, 'food': 1, 'art_music': 2, 'history': 3}
Precision: 0.5865
Recall: 0.2890
F1 Score: 0.1984
```

## 4. Использованные источники

Йылдырым C., Асгари-Ченаглу М. Осваиваем архитектуру Transformer // ДМК-Пресс, – 2022–320с.

Описание архитектуры трансформеров [электронный ресурс]. – Режим доступа: https://habr.com/ru/companies/mws/articles/770202/ (дата обращения: 27.12.2024).

Описание моделей ALBERT, RoBERTa, и DistilBERT [электронный ресурс]. – Режим доступа: https://habr.com/ru/articles/680986/ (дата обращения: 27.12.2024).

Документация PyTorch [Электронный ресурс]. – Режим доступа: https://pytorch.org/docs/stable/ (дата обращения: 27.12.2024).



