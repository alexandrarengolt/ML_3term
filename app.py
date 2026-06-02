import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import tensorflow as tf
import sys
import types
import sklearn
import sklearn.pipeline

# 1. Создаем универсальный динамический класс-пустышку
class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        # Если pickle ищет Pipeline, отдаем ему настоящий пайплайн из sklearn
        if name == 'Pipeline':
            return sklearn.pipeline.Pipeline
        # Для всех остальных классов (трансформеров PyCaret) возвращаем базовый класс-пустышку,
        # чтобы pickle мог воссоздать структуру без оригинального кода
        class GenericDummy:
            def __init__(self, *args, **kwargs): pass
            def __setstate__(self, state): self.__dict__.update(state)
        return GenericDummy

# 2. Регистрируем этот умный модуль во всех возможных путях импорта PyCaret
pycaret_dummy = DummyModule('pycaret')

sys.modules['pycaret'] = pycaret_dummy
sys.modules['pycaret.internal'] = pycaret_dummy
sys.modules['pycaret.internal.pipeline'] = pycaret_dummy
sys.modules['pycaret.internal.preprocess'] = pycaret_dummy
sys.modules['pycaret.internal.preprocess.transformers'] = pycaret_dummy
sys.modules['pycaret.internal.preprocess.iterative_imputer'] = pycaret_dummy

st.set_page_config(
    page_title="ML РГР",
    layout="wide"
)

#МЕНЮ
st.sidebar.title("Меню")
page = st.sidebar.radio(
    "Перейти на страницу:",
    ["Страница 1: Разработчик", 
     "Страница 2: Описание датасета", 
     "Страница 3: Визуализация зависимостей", 
     "Страница 4: Предсказание модели"]
)
@st.cache_resource
def load_real_model(model_key):
    model_mapping = {
        "ML1: Классическая модель": "polynom_regr.pkl",
        "ML2: Ансамблевая модель (Бэггинг)": "bagging_model.pkl",
        "ML3: Продвинутый бустинг (CatBoost)": "catboost_model.pkl",
        "ML4: Ансамблевая модель (Бустинг)": "boosting_model.pkl",
        "ML5: Ансамблевая модель (Стэкинг)": "stacking_model.pkl",
        "ML6: Глубокая нейросеть": "keras_tuner_model.keras"
    }
    
    file_name = model_mapping.get(model_key)
    
    if not file_name or not os.path.exists(file_name):
        return None
        
    if file_name.endswith('.keras'):
        return tf.keras.models.load_model(file_name)
    else:
        with open(file_name, "rb") as f:
            return pickle.load(f)
#СТРАНИЦА 1
if page == "Страница 1: Разработчик":
    st.title("Расчетно-графическая работа по машинному обучению")
    st.subheader("Информация о разработчике")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("lil_sonya.png", caption="Фото разработчика", width="stretch")
        
    with col2:
        st.markdown("""
        **ФИО:** Александра Ренгольт  
        **Номер учебной группы:** МО-241 \n
        **Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных """)

#СТРАНИЦА 2
elif page == "Страница 2: Описание датасета":
    st.title("Описание набора данных Air Quality")
    
    st.header("1. Предметная область")
    st.write("""Набор данных содержит ответы мультисенсорного устройства с 5 металлическими химическими датчиками оксидов. 
    Устройство находилось на уровне поля в загрязненном районе итальянского города. Данные собирались в течение года.""")
    
    st.header("2. Список полей и признаки")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Целевой признак")
        st.info("**CO(GT)** — Истинная среднечасовая концентрация CO в мг/м³ (эталон)")
        
        st.subheader("Временные метки")
        st.write("- **Date**: Дата (ДД/ММ/ГГГГ)")
        st.write("- **Time**: Время (ЧЧ.ММ.СС)")
        
        st.subheader("Внешние факторы")
        st.write("- **T**: Температура")
        st.write("- **RH**: Относительная влажность")
        st.write("- **AH**: Абсолютная влажность")

    with col2:
        st.subheader("Показания датчиков")
        st.write("- **PT08.S1(CO)**: Оксид углерода")
        st.write("- **PT08.S2(NMHC)**: Неметаллические углеводороды")
        st.write("- **PT08.S3(NOx)**: Оксиды азота")
        st.write("- **PT08.S4(NO2)**: Диоксид азота")
        st.write("- **PT08.S5(O3)**: Озон")
        
        st.subheader("Эталонные значения (GT)")
        st.write("- **NMHC(GT)**: Неметаллические углеводороды (мкг/м³)")
        st.write("- **C6H6(GT)**: Бензол (мкг/м³)")
        st.write("- **NOx(GT)**: Оксиды азота (ppb)")
        st.write("- **NO2(GT)**: Диоксид азота (мкг/м³)")

    st.header("3. Задача")
    st.success("Тип задачи: **Регрессия**. Необходимо предсказать уровень CO(GT) на основе данных сенсоров и метеоусловий.")
    
    st.header("4. Особенности предобработки")
    st.warning("""**Важно:** В данном датасете отсутствующие значения помечены числом **-200**.Перед обучением необходимо заменить их на средние значения или использовать интерполяцию. """)


#СТРАНИЦА 3
elif page == "Страница 3: Визуализация зависимостей":
    st.title("Визуализация зависимостей в наборе данных")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Связь между целевым признаком и сенсорами")
        st.image("plot1.png", width="stretch")

        st.subheader("Матрица корреляции")
        st.image("matrix.png", width="stretch")


    with col2:
        st.subheader("Анализ выбросов и диапазонов (Box Plot)")
        st.image("boxplot.png",  width="stretch")


        st.subheader("Концентрация CO(GT) по часам")
        st.image("plot3.png",  width="stretch")


#СТРАНИЦА 4
elif page == "Страница 4: Предсказание модели":
    st.title("Получение предсказаний модели")
    st.markdown("---")

    # Выбор модели пользователем
    model_choice = st.selectbox(
        "Выберите модель ML для инференса:",
        [
            "ML1: Классическая модель",
            "ML2: Ансамблевая модель (Бэггинг)",
            "ML3: Продвинутый бустинг (CatBoost)",
            "ML4: Ансамблевая модель (Бустинг)",
            "ML5: Ансамблевая модель (Стэкинг)",
            "ML6: Глубокая нейросеть"
        ]
    )

    # Загрузка выбранной модели
    model = load_real_model(model_choice)

    if model is None:
        st.error(f"Файл модели для `{model_choice}` не найден в директории проекта. Проверьте репозиторий GitHub.")
    else:
        st.success(f"Модель `{model_choice}` успешно загружена и готова к работе.")
        st.markdown("### Данные для одиночного инференса")

        # Отрисовка интерфейса ввода параметров в 3 колонки
        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            st.markdown("##### Показания сенсоров")
            s1 = st.number_input("PT08.S1(CO)", min_value=0.0, max_value=3000.0, value=1000.0)
            s2 = st.number_input("PT08.S2(NMHC)", min_value=0.0, max_value=3000.0, value=900.0)
            s3 = st.number_input("PT08.S3(NOx)", min_value=0.0, max_value=3000.0, value=800.0)
            s4 = st.number_input("PT08.S4(NO2)", min_value=0.0, max_value=3000.0, value=1400.0)
            s5 = st.number_input("PT08.S5(O3)", min_value=0.0, max_value=3000.0, value=1000.0)

        with col_input2:
            st.markdown("##### Параметры газов и времени")
            c6h6 = st.slider("C6H6(GT) (Бензол)", min_value=0.0, max_value=100.0, value=10.0)
            nox = st.number_input("NOx(GT)", min_value=0.0, max_value=2000.0, value=200.0)
            no2 = st.number_input("NO2(GT)", min_value=0.0, max_value=500.0, value=100.0)
            
            chosen_date = st.date_input("Выберите дату:")
            chosen_hour = st.slider("Выберите час (hour):", min_value=0, max_value=23, value=12)

        with col_input3:
            st.markdown("##### Метеоусловия")
            temp = st.slider("T (Температура, °C)", min_value=-10.0, max_value=50.0, value=20.0)
            rh = st.slider("RH (Относительная влажность, %)", min_value=0.0, max_value=100.0, value=50.0)
            ah = st.slider("AH (Абсолютная влажность)", min_value=0.0, max_value=3.0, value=1.0, step=0.0001, format="%.4f")

        st.markdown("---")

        # КНОПКА ЗАПУСКА ОДИНОЧНОГО РАСЧЕТА
        if st.button("Рассчитать концентрацию CO(GT)", type="primary"):
            # 1. Расчет временных фичей на лету (Feature Engineering)
            year = chosen_date.year
            month = chosen_date.month
            day = chosen_date.day
            weekday = chosen_date.weekday()
            is_weekend = 1 if weekday >= 5 else 0

            # 2. Строгий One-Hot Encoding для дней недели с префиксами "Day_" (как на обучении)
            days_ohe = {
                "Day_Monday": 0, "Day_Tuesday": 0, "Day_Wednesday": 0, 
                "Day_Thursday": 0, "Day_Friday": 0, "Day_Saturday": 0, "Day_Sunday": 0
            }
            days_mapping = ["Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday", "Day_Friday", "Day_Saturday", "Day_Sunday"]
            days_ohe[days_mapping[weekday]] = 1

            # 3. Базовый набор из 11 признаков (для нейросети ML6)
            base_11_features = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]

            # 4. Расширенный набор из 23 признаков со строгими именами (для классики и бустингов)
            extended_columns = base_11_features + [
                "year", "month", "Day", "hour", "is_weekend", 
                "Day_Monday", "Day_Tuesday", "Day_Wednesday", 
                "Day_Thursday", "Day_Friday", "Day_Saturday", "Day_Sunday"
            ]

            df_11 = pd.DataFrame([[s1, c6h6, s2, nox, s3, no2, s4, s5, temp, rh, ah]], columns=base_11_features)
            
            df_23 = pd.DataFrame([[
                s1, c6h6, s2, nox, s3, no2, s4, s5, temp, rh, ah,
                year, month, day, chosen_hour, is_weekend,
                days_ohe["Day_Monday"], days_ohe["Day_Tuesday"], days_ohe["Day_Wednesday"],
                days_ohe["Day_Thursday"], days_ohe["Day_Friday"], days_ohe["Day_Saturday"], days_ohe["Day_Sunday"]
            ]], columns=extended_columns)

            # Датафрейм-заглушка с таргетом для старых PyCaret моделей
            df_pycaret = df_23.copy()
            df_pycaret["CO(GT)"] = 0.0

            # 5. Интеллектуальный каскадный блок инференса (фикс багов ndarray и CatBoost)
            prediction = None
            try:
                if "ML6" in model_choice:  # Глубокая нейросеть Keras/TensorFlow
                    expected_features = model.input_shape[1]
                    features_matrix = df_11.to_numpy(dtype=np.float32) if expected_features == 11 else df_23.to_numpy(dtype=np.float32)
                    raw_res = model.predict(features_matrix, verbose=0)
                    prediction = raw_res.flatten()[0]
                else:  # Классические модели, CatBoost, Ансамбли
                    try:
                        raw_res = model.predict(df_23)
                        prediction = raw_res[0] if isinstance(raw_res, np.ndarray) else raw_res
                    except ValueError:
                        try:
                            raw_res = model.predict(df_pycaret)
                            prediction = raw_res[0] if isinstance(raw_res, np.ndarray) else raw_res
                        except ValueError:
                            raw_res = model.predict(df_11)
                            prediction = raw_res[0] if isinstance(raw_res, np.ndarray) else raw_res

                # Финальная распаковка, если на выходе остался вложенный массив
                if isinstance(prediction, (np.ndarray, list)):
                    prediction = float(np.array(prediction).flatten()[0])
                else:
                    prediction = float(prediction)

            except Exception as e:
                st.error(f"Ошибка вызова математического ядра: {e}")
                prediction = None

            # 6. Вывод результатов расчета с экологическим зонированием
            if prediction is not None:
                st.markdown("### Результат расчета:")
                st.metric(label="Прогнозируемая концентрация CO(GT)", value=f"{prediction:.2f} мг/м³")

                if prediction < 1.5:
                    st.success("Уровень воздуха: Безопасный (Концентрация в пределах нормы)")
                elif 1.5 <= prediction < 3.5:
                    st.warning("Уровень воздуха: Умеренное загрязнение (Рекомендуется проветривание)")
                else:
                    st.error("Уровень воздуха: ОПАСНО! (Зафиксирован критический выброс CO)")

        # ==========================================================
        # БЛОК 2: ПАКЕТНЫЙ ИНФЕРЕНС ИЗ ФАЙЛОВ (.CSV)
        # ==========================================================
        # ==========================================================
        # БЛОК 2: ПАКЕТНЫЙ ИНФЕРЕНС ИЗ ФАЙЛОВ (.CSV)
        # ==========================================================
        st.markdown("---")
        st.markdown("### Пакетный инференс (Обработка массивов данных)")
        
        data_source = st.radio(
            "Выберите источник данных для пакетного расчета:",
            ["Использовать демонстрационный датасет", "Загрузить свой файл (.csv)"]
        )

        user_df = None

        if data_source == "Использовать демонстрационный датасет":
            demo_file_path = "X_test.csv" 
            if os.path.exists(demo_file_path):
                user_df = pd.read_csv(demo_file_path)
                st.info("Загружен демонстрационный датасет из репозитория.")
            else:
                st.warning(f"Файл `{demo_file_path}` не найден в репозитории. Пожалуйста, загрузите его вручную.")
        
        elif data_source == "Загрузить свой файл (.csv)":
            uploaded_file = st.file_uploader("Выберите файл данных в формате *.csv для массового расчета:", type="csv")
            if uploaded_file is not None:
                user_df = pd.read_csv(uploaded_file)

        if user_df is not None:
            st.markdown("**Фрагмент набора данных (Первые 5 строк):**")
            st.dataframe(user_df.head(5))

            if st.button("Запустить пакетный расчет всего файла", type="secondary"):
                base_11_features = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
                
                # Проверяем наличие 11 базовых колонок
                missing_cols = [col for col in base_11_features if col not in user_df.columns]
                
                if not missing_cols:
                    try:
                        # Автоматический Feature Engineering для пакета данных
                        batch_df = user_df.copy()
                        
                        # Если в датасете нет временных колонок, создаем дефолтные (как в интерфейсе)
                        if "year" not in batch_df.columns:
                            batch_df["year"] = chosen_date.year
                            batch_df["month"] = chosen_date.month
                            batch_df["Day"] = chosen_date.day
                            batch_df["hour"] = chosen_hour
                            weekday = chosen_date.weekday()
                            batch_df["is_weekend"] = 1 if weekday >= 5 else 0
                            
                            # Генерируем One-Hot Encoding для дней недели со строгими префиксами "Day_"
                            days_mapping = ["Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday", "Day_Friday", "Day_Saturday", "Day_Sunday"]
                            for day_col in days_mapping:
                                batch_df[day_col] = 0
                            batch_df[days_mapping[weekday]] = 1

                        # Формируем матрицы нужной геометрии для проверки моделей
                        extended_columns = base_11_features + [
                            "year", "month", "Day", "hour", "is_weekend", 
                            "Day_Monday", "Day_Tuesday", "Day_Wednesday", 
                            "Day_Thursday", "Day_Friday", "Day_Saturday", "Day_Sunday"
                        ]
                        
                        df_batch_11 = batch_df[base_11_features]
                        df_batch_23 = batch_df[extended_columns]
                        df_batch_pycaret = df_batch_23.copy()
                        df_batch_pycaret["CO(GT)"] = 0.0

                        # Безопасный инференс для пакета данных
                        if "ML6" in model_choice:  # Нейросеть
                            expected_features = model.input_shape[1]
                            features_matrix = df_batch_11.to_numpy(dtype=np.float32) if expected_features == 11 else df_batch_23.to_numpy(dtype=np.float32)
                            raw_preds = model.predict(features_matrix, verbose=0)
                            predictions_array = raw_preds.flatten()
                        else:  # Классика, CatBoost, Ансамбли
                            try:
                                raw_preds = model.predict(df_batch_23)
                            except ValueError:
                                try:
                                    raw_preds = model.predict(df_batch_pycaret)
                                except ValueError:
                                    raw_preds = model.predict(df_batch_11)
                            
                            # Фикс ошибки 'numpy.ndarray' object has no attribute 'predict'
                            # Извлекаем одномерный массив, если модель вернула матрицу или сложный объект
                            if hasattr(raw_preds, "flatten"):
                                predictions_array = raw_preds.flatten()
                            else:
                                predictions_array = np.array(raw_preds).flatten()

                        # Интеграция результатов в итоговую таблицу
                        result_df = user_df.copy()
                        result_df["Predicted_CO(GT)"] = predictions_array

                        st.success(f"Массовый расчет завершен! Успешно обработано строк: {len(result_df)}")
                        st.dataframe(result_df.head(10))

                        # Генерация кнопки скачивания
                        csv_buffer = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать файл с результатами предсказаний",
                            data=csv_buffer,
                            file_name="air_quality_predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as batch_error:
                        st.error(f"Ошибка при обработке массива данных: {batch_error}")
                else:
                    st.error(f"В загруженном датасете отсутствуют необходимые колонки факторов: {missing_cols}")
