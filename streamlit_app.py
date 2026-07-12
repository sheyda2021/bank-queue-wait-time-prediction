
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# تنظیمات صفحه
# ------------------------------------------------------------
st.set_page_config(
    page_title="پیش‌بینی زمان انتظار مشتری بانک",
    page_icon="🏦",
    layout="centered"
)


OUTPUT_DIR = "bank_simulation_outputs"

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_rf.joblib")
FEATURE_COLUMNS_PATH = os.path.join(OUTPUT_DIR, "feature_columns.joblib")
SERVICE_TYPES_PATH = os.path.join(OUTPUT_DIR, "service_types.joblib")


@st.cache_resource
def load_artifacts():
"
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    service_types = joblib.load(SERVICE_TYPES_PATH)
    return model, feature_columns, service_types


model, feature_columns, service_types = load_artifacts()

# ------------------------------------------------------------
# عنوان و توضیح
# ------------------------------------------------------------
st.title("🏦 پیش‌بینی زمان انتظار مشتری")

if model is None:
    st.error(
        "فایل مدل پیدا نشد. ابتدا باید فایل اصلی پایتون "
        f"(که شامل شبیه‌سازی و آموزش مدل است) را اجرا کنی تا فایل‌های "
        f"مدل در پوشه‌ی «{OUTPUT_DIR}» ساخته شوند. سپس این داشبورد را "
        "دوباره اجرا کن."
    )
    st.stop()

st.divider()

# ------------------------------------------------------------
# فرم ورودی کاربر
# ------------------------------------------------------------
st.subheader("ورودی ویژگی‌ها")

col1, col2 = st.columns(2)

with col1:
    num_counters = st.selectbox(
        "تعداد باجه‌های فعال",
        options=[3, 4, 5],
        index=0,
        help="مدل فقط با این سه مقدار آموزش دیده است."
    )

    service_type = st.selectbox(
        "نوع خدمت درخواستی",
        options=service_types,
        index=0
    )

    queue_length_at_arrival = st.number_input(
        "طول صف در لحظه ورود مشتری (تعداد نفرات منتظر)",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
        help="مهم‌ترین ویژگی برای پیش‌بینی، طبق تحلیل Feature Importance."
    )

with col2:
    arrival_time = st.number_input(
        "زمان ورود مشتری (دقیقه از ابتدای شبیه‌سازی)",
        min_value=0.0,
        max_value=1000.0,
        value=60.0,
        step=1.0
    )

    service_time = st.number_input(
        "زمان خدمت تخمینی (دقیقه)",
        min_value=0.1,
        max_value=120.0,
        value=5.0,
        step=0.5,
        help="اگر نمی‌دانی، می‌توانی میانگین زمان خدمت همان نوع خدمت را وارد کنی."
    )

st.divider()


input_dict = {
    'num_counters': num_counters,
    'arrival_time': arrival_time,
    'service_time': service_time,
    'queue_length_at_arrival': queue_length_at_arrival,
}

# ستون‌های one-hot برای نوع خدمت؛ همه صفر مگر نوع خدمت انتخاب‌شده
for col in feature_columns:
    if col.startswith('service_type_'):
        selected_service_col = f'service_type_{service_type}'
        input_dict[col] = 1 if col == selected_service_col else 0

# تبدیل به DataFrame با ترتیب دقیق ستون‌های آموزش
input_df = pd.DataFrame([input_dict])[feature_columns]

with st.expander("نمایش بردار ورودی نهایی (برای بررسی/دیباگ)"):
    st.dataframe(input_df)


# پیش‌بینی
if st.button("پیش‌بینی زمان انتظار", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    prediction = max(0.0, prediction)  # زمان انتظار منفی فیزیکی نیست

    st.success(f"⏱️ زمان انتظار پیش‌بینی‌شده: **{prediction:.2f} دقیقه**")

    # توضیح کیفی برای کمک به ارائه: مقایسه با حد آستانه‌های تجربی
    if prediction < 5:
        st.info("این مقدار نسبتاً کم است؛ صف در وضعیت مناسبی قرار دارد.")
    elif prediction < 15:
        st.warning("زمان انتظار متوسط است؛ ممکن است نیاز به بررسی بیشتر باشد.")
    else:
        st.error("زمان انتظار بالاست؛ افزایش تعداد باجه‌ها می‌تواند مؤثر باشد.")

st.divider()
st.caption(
    "مدل: Random Forest Regressor | تیون‌شده با Optuna | "
    "ویژگی مهم‌ترین: طول صف در لحظه ورود (طبق تحلیل Feature Importance و SHAP)"
)
