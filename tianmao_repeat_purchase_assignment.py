import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import gc # Garbage collection
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import entropy
import re
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# --- 0. 全局配置与实验开关 ---
print("--- 0. 全局配置与实验开关 ---")
DATA_PATH = './data_format1/'
FILE_USER_LOG_PART = DATA_PATH + 'user_log_format1.csv'
FILE_USER_INFO = DATA_PATH + 'user_info_format1.csv'
FILE_TRAIN_ORIG = DATA_PATH + 'train_format1.csv'
FILE_TEST_ORIG = DATA_PATH + 'test_format1.csv'

# --- 实验配置开关 ---
# 特征工程增强开关
ADD_TIME_WINDOW_FEATURES = True
ADD_COMPLEX_CONVERSION_RATES = True
ADD_BUY_INTERVAL_FEATURES = True
ADD_ENTROPY_FEATURES = True

# 观察期处理开关
PROCESS_OBSERVATION_PERIOD = True
FILTER_SHORT_OBSERVATION_USERS = True # True: 移除观察期不足用户; False: 保留并添加指示特征

# 数据变换开关
APPLY_FEATURE_BINNING = True
APPLY_LOG_TRANSFORM = True

# 特征选择开关
APPLY_VARIANCE_THRESHOLD = True
APPLY_SELECTKBEST = True
APPLY_RFE = False # 非常耗时，默认关闭
DO_POST_CV_FEATURE_SELECTION = True

# 类别不平衡处理策略
IMBALANCE_STRATEGY = "scale_pos_weight" # "smote", "random_undersample", "scale_pos_weight", "none"

# 超参数调优开关
DO_HYPERPARAM_TUNING = False # 非常耗时，默认关闭
TUNING_METHOD = "RandomizedSearch" # "GridSearch", "RandomizedSearch"

# --- 1. 数据加载与初步探索 ---
print("--- 1. 数据加载与初步探索 ---")
try:
    user_log_df = pd.read_csv(FILE_USER_LOG_PART)
    user_info_df = pd.read_csv(FILE_USER_INFO)
    train_df_orig = pd.read_csv(FILE_TRAIN_ORIG)
    test_df_orig = pd.read_csv(FILE_TEST_ORIG)
except FileNotFoundError as e:
    print(f"错误：找不到数据文件，请检查路径配置。{e}")
    exit()

# 新增：检查原始表的重复性
print("\n--- 1.1 检查原始表重复性 ---")
print(f"原始训练集 train_df_orig 中重复行数: {train_df_orig.duplicated().sum()}")
print(f"原始测试集 test_df_orig 中重复行数: {test_df_orig.duplicated().sum()}")
# user_info 表中 user_id 应唯一
if 'user_id' in user_info_df.columns:
    print(f"用户信息表 user_info_df 中 user_id 重复数: {user_info_df.duplicated(subset=['user_id']).sum()}")
    if user_info_df.duplicated(subset=['user_id']).sum() > 0:
        user_info_df = user_info_df.drop_duplicates(subset=['user_id'], keep='first')
        print("已移除 user_info_df 中的重复user_id。")
else:
    print("警告: user_info_df 中未找到 'user_id' 列。")


print(f"训练集标签分布:\n{train_df_orig['label'].value_counts(normalize=True)}")


# --- 2. 数据预处理与清洗 ---
print("\n--- 2. 数据预处理与清洗 ---")
# 2.1 user_info_df 清洗
user_info_df['age_range'] = user_info_df['age_range'].fillna(0)
user_info_df['age_range'] = user_info_df['age_range'].replace({7: 6, 8: 6})
user_info_df['gender'] = user_info_df['gender'].fillna(2)

# 2.2 user_log_df 清洗与准备
if 'seller_id' in user_log_df.columns:
    user_log_df.rename(columns={'seller_id': 'merchant_id'}, inplace=True)
user_log_df['brand_id'] = user_log_df['brand_id'].fillna(0)

days_in_month = {5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30}
month_start_day = {5:1}
for m in range(6, 12): month_start_day[m] = month_start_day[m-1] + days_in_month[m-1]
def mmdd_to_day_number(mmdd):
    if pd.isna(mmdd) or mmdd == 0: return np.nan
    month, day = int(mmdd // 100), int(mmdd % 100)
    # 确保月份在定义的字典中，否则返回NaN
    base_day = month_start_day.get(month)
    if base_day is None: return np.nan
    return base_day + day -1
user_log_df['abs_day'] = user_log_df['time_stamp'].apply(mmdd_to_day_number)
user_log_df['month'] = user_log_df['time_stamp'] // 100
user_log_df.dropna(subset=['abs_day'], inplace=True) # 重要：移除无法计算abs_day的行

def reduce_mem_usage(df, verbose=True):
    # ... (内容同前)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min) or pd.isna(c_max): continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

print("优化内存...")
user_log_df = reduce_mem_usage(user_log_df)
user_info_df = reduce_mem_usage(user_info_df)
train_df_orig = reduce_mem_usage(train_df_orig, verbose=False)
test_df_orig = reduce_mem_usage(test_df_orig, verbose=False)

# 2.3 合并数据
train_df = pd.merge(train_df_orig, user_info_df, on='user_id', how='left')
test_df = pd.merge(test_df_orig, user_info_df, on='user_id', how='left')
train_df['origin'] = 'train'; test_df['origin'] = 'test'
all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
all_df.drop(columns=['prob'], inplace=True, errors='ignore')

# --- 3. 特征工程 (实现所有讨论的思路) ---
print("\n--- 3. 特征工程 (完整增强版) ---")
# 关于 'action_type_path' 或 'time_stamp_path'：
# 当前特征工程主要基于聚合统计量，未使用原始序列路径作为直接特征。
# 时间信息通过 'abs_day' 参与计算时间差、首次/末次活动、时间窗口等。

# Helper for time differences (改进NaN填充思路)
def get_time_diff_stats(series_abs_day):
    series_abs_day = series_abs_day.dropna().sort_values()
    if len(series_abs_day) < 2:
        # 返回NaN，后续统一处理。0可能是一个有效的差值。
        return pd.Series({
            'mean_time_diff': np.nan, 'median_time_diff': np.nan, 'std_time_diff': np.nan,
            'min_time_diff': np.nan, 'max_time_diff': np.nan
        })
    diffs = series_abs_day.diff().dropna()
    if diffs.empty:
        return pd.Series({
            'mean_time_diff': np.nan, 'median_time_diff': np.nan, 'std_time_diff': np.nan,
            'min_time_diff': np.nan, 'max_time_diff': np.nan
        })
    return pd.Series({
        'mean_time_diff': diffs.mean(), 'median_time_diff': diffs.median(), 'std_time_diff': diffs.std(),
        'min_time_diff': diffs.min(), 'max_time_diff': diffs.max()
    })

# 3.2 用户整体行为特征
print("衍生用户整体行为特征...")
user_behavior_features_agg = {
    'user_log_count': ('item_id', 'count'), 'user_item_nunique': ('item_id', 'nunique'),
    'user_cat_nunique': ('cat_id', 'nunique'), 'user_merchant_nunique': ('merchant_id', 'nunique'),
    'user_brand_nunique': ('brand_id', lambda x: x[x != 0].nunique()),
    'user_time_stamp_nunique': ('time_stamp', 'nunique'), 'user_month_nunique': ('month', 'nunique'),
    'user_action_type_0_count': ('action_type', lambda x: (x==0).sum()), 'user_action_type_1_count': ('action_type', lambda x: (x==1).sum()),
    'user_action_type_2_count': ('action_type', lambda x: (x==2).sum()), 'user_action_type_3_count': ('action_type', lambda x: (x==3).sum()),
    'user_first_abs_day': ('abs_day', 'min'), 'user_last_abs_day': ('abs_day', 'max'),
}
user_behavior_features = user_log_df.groupby('user_id').agg(**user_behavior_features_agg).reset_index()
user_behavior_features['user_avg_logs_per_active_day'] = user_behavior_features['user_log_count'] / (user_behavior_features['user_time_stamp_nunique'] + 1e-6)
for i in range(4): user_behavior_features[f'user_action_type_{i}_ratio'] = user_behavior_features[f'user_action_type_{i}_count'] / (user_behavior_features['user_log_count'] + 1e-6)

if ADD_BUY_INTERVAL_FEATURES:
    print("Calculating user buy interval features...")
    purchase_logs = user_log_df[user_log_df['action_type'] == 2].sort_values(by=['user_id', 'abs_day'])
    if not purchase_logs.empty:
        purchase_logs['buy_interval'] = purchase_logs.groupby('user_id')['abs_day'].diff()
        user_buy_interval_stats = purchase_logs.groupby('user_id')['buy_interval'].agg(
            u_mean_buy_interval='mean', u_median_buy_interval='median', u_std_buy_interval='std'
        ).reset_index()
        user_behavior_features = pd.merge(user_behavior_features, user_buy_interval_stats, on='user_id', how='left')
    else: # No purchase logs at all
        for col in ['u_mean_buy_interval', 'u_median_buy_interval', 'u_std_buy_interval']: user_behavior_features[col] = np.nan


if ADD_COMPLEX_CONVERSION_RATES:
    user_behavior_features['user_click_to_addcart_ratio'] = user_behavior_features['user_action_type_1_count'] / (user_behavior_features['user_action_type_0_count'] + 1e-6)
    user_behavior_features['user_addcart_to_buy_ratio'] = user_behavior_features['user_action_type_2_count'] / (user_behavior_features['user_action_type_1_count'] + 1e-6)
    user_behavior_features['user_fav_to_buy_ratio'] = user_behavior_features['user_action_type_2_count'] / (user_behavior_features['user_action_type_3_count'] + 1e-6)
if ADD_ENTROPY_FEATURES:
    print("Calculating user entropy features...")
    if not user_log_df.empty:
        user_cat_entropy = user_log_df.groupby('user_id')['cat_id'].apply(lambda x: entropy(x.value_counts(normalize=True)) if not x.empty else np.nan).reset_index().rename(columns={'cat_id':'user_cat_entropy'})
        user_behavior_features = pd.merge(user_behavior_features, user_cat_entropy, on='user_id', how='left')
    else: user_behavior_features['user_cat_entropy'] = np.nan

print("Calculating user time difference features (may take time)...")
if not user_log_df.empty:
    user_time_diffs = user_log_df.groupby('user_id')['abs_day'].apply(get_time_diff_stats).reset_index()
    user_time_diffs = user_time_diffs.rename(columns=lambda c: 'u_' + c if c not in ['user_id'] else c)
    user_behavior_features = pd.merge(user_behavior_features, user_time_diffs, on='user_id', how='left')
else: # Create empty columns if user_log_df is empty
    for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']:
        user_behavior_features[f'u_{col_suffix}'] = np.nan

all_df = pd.merge(all_df, user_behavior_features, on='user_id', how='left'); gc.collect()

# 3.3 用户-商家互动特征
print("衍生用户-商家互动特征...")
um_interaction_agg = {
    'um_log_count':('item_id','count'), 'um_item_nunique':('item_id','nunique'),
    'um_cat_nunique':('cat_id','nunique'), 'um_brand_nunique':('brand_id',lambda x: x[x!=0].nunique()),
    'um_time_stamp_nunique':('time_stamp','nunique'),
    'um_action_type_0_count':('action_type',lambda x:(x==0).sum()), 'um_action_type_1_count':('action_type',lambda x:(x==1).sum()),
    'um_action_type_2_count':('action_type',lambda x:(x==2).sum()), 'um_action_type_3_count':('action_type',lambda x:(x==3).sum()),
    'um_first_abs_day':('abs_day','min'), 'um_last_abs_day':('abs_day','max'),
}
user_merchant_interaction = user_log_df.groupby(['user_id', 'merchant_id']).agg(**um_interaction_agg).reset_index()
for i in range(4): user_merchant_interaction[f'um_action_type_{i}_ratio'] = user_merchant_interaction[f'um_action_type_{i}_count'] / (user_merchant_interaction['um_log_count'] + 1e-6)
print("Calculating user-merchant time difference features (may take time)...")
if not user_log_df.empty:
    um_time_diffs = user_log_df.groupby(['user_id', 'merchant_id'])['abs_day'].apply(get_time_diff_stats).reset_index()
    um_time_diffs = um_time_diffs.rename(columns=lambda c: 'um_' + c if c not in ['user_id', 'merchant_id'] else c)
    user_merchant_interaction = pd.merge(user_merchant_interaction, um_time_diffs, on=['user_id', 'merchant_id'], how='left')
else: # Create empty columns
    for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']:
        user_merchant_interaction[f'um_{col_suffix}'] = np.nan


all_df = pd.merge(all_df, user_merchant_interaction, on=['user_id', 'merchant_id'], how='left')
if 'um_time_stamp_nunique' in all_df.columns and 'user_time_stamp_nunique' in all_df.columns:
    all_df['um_active_days_ratio_in_user'] = all_df['um_time_stamp_nunique'] / (all_df['user_time_stamp_nunique'] + 1e-6)
else: all_df['um_active_days_ratio_in_user'] = 0 # Or np.nan if preferred
gc.collect()

# 3.3.1 时间窗口特征
if ADD_TIME_WINDOW_FEATURES:
    if not user_log_df['abs_day'].empty: # Check if abs_day has valid values
        max_abs_day_in_log = user_log_df['abs_day'].max()
        for T_window in [7, 15, 30]:
            print(f"Generating features for last {T_window} days...")
            window_log_df = user_log_df[user_log_df['abs_day'] > (max_abs_day_in_log - T_window)]
            if not window_log_df.empty:
                user_recent_agg_df = window_log_df.groupby('user_id').agg(
                    **{f'u_logs_last_{T_window}d': ('item_id', 'count'),
                       f'u_buys_last_{T_window}d': ('action_type', lambda x: (x == 2).sum()),
                       f'u_cats_last_{T_window}d': ('cat_id', 'nunique'),
                       f'u_items_last_{T_window}d': ('item_id', 'nunique'),
                       f'u_active_days_last_{T_window}d': ('abs_day', 'nunique')}
                ).reset_index()
                all_df = pd.merge(all_df, user_recent_agg_df, on='user_id', how='left')
                um_recent_agg_df = window_log_df.groupby(['user_id', 'merchant_id']).agg(
                    **{f'um_logs_last_{T_window}d': ('item_id', 'count'),
                       f'um_buys_last_{T_window}d': ('action_type', lambda x: (x == 2).sum())}
                ).reset_index()
                all_df = pd.merge(all_df, um_recent_agg_df, on=['user_id', 'merchant_id'], how='left')
            else: # If window_log_df is empty, create placeholder columns with NaN or 0
                for prefix in ['u', 'um']:
                    all_df[f'{prefix}_logs_last_{T_window}d'] = 0
                    all_df[f'{prefix}_buys_last_{T_window}d'] = 0
                    if prefix == 'u':
                        all_df[f'{prefix}_cats_last_{T_window}d'] = 0
                        all_df[f'{prefix}_items_last_{T_window}d'] = 0
                        all_df[f'{prefix}_active_days_last_{T_window}d'] = 0
            gc.collect()
    else: print("无法计算时间窗口特征，'abs_day' 为空或无效。")


# 3.4 商家自身特征
print("衍生商家特征...")
merchant_features_agg = {
    'm_user_nunique':('user_id','nunique'), 'm_log_count':('item_id','count'),
    'm_item_nunique':('item_id','nunique'), 'm_cat_nunique':('cat_id','nunique'),
    'm_brand_nunique':('brand_id',lambda x:x[x!=0].nunique()),
    'm_buy_count':('action_type',lambda x:(x==2).sum()),
    'm_buy_user_nunique':('user_id',lambda x: x[user_log_df.loc[x.index,'action_type']==2].nunique() if not x.empty else 0) # Handle empty group
}
merchant_features = user_log_df.groupby('merchant_id').agg(**merchant_features_agg).reset_index()
merchant_features['m_buyer_conversion_rate'] = merchant_features['m_buy_user_nunique'] / (merchant_features['m_user_nunique'] + 1e-6)
all_df = pd.merge(all_df, merchant_features, on='merchant_id', how='left'); gc.collect()

# 3.5 处理特征工程产生的缺失值 (改进版)
print("填充特征工程产生的缺失值...")
keywords_for_nan_fill = ['count', 'nunique', 'ratio', 'entropy', '_last_', '_buy_', '_log_count', 'active_days']
# Time diff / interval features should be handled differently
time_related_nan_cols = [col for col in all_df.columns if 'time_diff' in col or 'interval' in col or 'abs_day' in col] # first/last abs_day

for col in all_df.columns:
    if col in ['user_id', 'merchant_id', 'label', 'origin', 'age_range', 'gender', 'time_stamp']: # Skip IDs, label, origin, and base already handled
        continue
    if any(keyword in col for keyword in keywords_for_nan_fill):
        all_df[col] = all_df[col].fillna(0) # Counts, ratios, entropies, last_X_days usually mean 0 if no activity
    elif col in time_related_nan_cols:
        # For time differences, first/last day: filling with 0 might be misleading if 0 is a valid value.
        # Median of the column, or a specific placeholder like -1 or a value outside typical range might be better.
        # For now, if it's a 'diff' or 'interval', fill with median, else with 0 (e.g. for first/last_abs_day if user had no logs)
        if 'diff' in col or 'interval' in col:
            median_val = all_df[col].median()
            all_df[col] = all_df[col].fillna(median_val if not pd.isna(median_val) else 0) # Fallback to 0 if median is NaN
        else: # first_abs_day, last_abs_day
            all_df[col] = all_df[col].fillna(0) # Or a distinct placeholder if 0 is a meaningful day number
    elif pd.api.types.is_numeric_dtype(all_df[col]): # General numeric, fill with 0 as a default
         all_df[col] = all_df[col].fillna(0)
    # Object columns (if any new ones created) should be checked and handled specifically

all_df['age_range'] = all_df['age_range'].fillna(0)
all_df['gender'] = all_df['gender'].fillna(2)
print("特征工程后 all_df 预览 (部分):")
print(all_df.sample(5)); all_df.info(verbose=True, show_counts=True); gc.collect()

# 3.6 处理训练集观察期差异
if PROCESS_OBSERVATION_PERIOD:
    # ... (逻辑同前一版，使用 PROCESS_OBSERVATION_PERIOD 和 FILTER_SHORT_OBSERVATION_USERS 控制) ...
    print("\n--- 3.6 处理训练集观察期差异 ---")
    if 'user_first_abs_day' in all_df.columns and not user_log_df['abs_day'].empty: # Check if 'user_first_abs_day' exists and logs were processed
        min_log_day_overall = user_log_df['abs_day'].min()
        grace_period_days = 30 
        observation_first_day_threshold = min_log_day_overall + grace_period_days
        original_train_count = all_df[all_df['origin'] == 'train'].shape[0]
        # Ensure user_first_abs_day is numeric and not NaN before comparison
        all_df['user_first_abs_day_numeric'] = pd.to_numeric(all_df['user_first_abs_day'], errors='coerce')

        condition_short_obs = (
            (all_df['origin'] == 'train') & 
            (all_df['user_first_abs_day_numeric'] > observation_first_day_threshold) & 
            (all_df['user_first_abs_day_numeric'].notna())
        )
        all_df['is_short_observation_train_user'] = 0
        all_df.loc[condition_short_obs, 'is_short_observation_train_user'] = 1
        print(f"被标记为潜在观察期不足的训练样本数: {all_df['is_short_observation_train_user'].sum()}")
        if FILTER_SHORT_OBSERVATION_USERS:
            print("筛选策略：移除观察期不足的训练用户。")
            all_df = all_df[~((all_df['origin'] == 'train') & (all_df['is_short_observation_train_user'] == 1))].copy()
            print(f"筛选后，训练集样本数: {all_df[all_df['origin'] == 'train'].shape[0]}")
        # Drop the temporary numeric column
        all_df.drop(columns=['user_first_abs_day_numeric'], inplace=True, errors='ignore')
        # If not filtering, 'is_short_observation_train_user' remains as a feature.
        # If filtering, this column is now mostly irrelevant for the filtered train set.
        # It will still exist for test set (all 0s), might be dropped later if not useful.
        if FILTER_SHORT_OBSERVATION_USERS and 'is_short_observation_train_user' in all_df.columns:
            all_df.drop(columns=['is_short_observation_train_user'], inplace=True, errors='ignore')

    else: print("警告: 'user_first_abs_day' 特征缺失或日志数据为空，无法执行观察期筛选。")


# --- 4. 特征变换与选择 ---
print("\n--- 4. 特征变换与选择 ---")
categorical_feats_lgbm = ['age_range', 'gender']
if not FILTER_SHORT_OBSERVATION_USERS and PROCESS_OBSERVATION_PERIOD and 'is_short_observation_train_user' in all_df.columns:
    if 'is_short_observation_train_user' not in categorical_feats_lgbm:
        categorical_feats_lgbm.append('is_short_observation_train_user')


# 4.A 数据变换 (分箱, log变换)
# ... (分箱和log变换逻辑同前一版，由 APPLY_FEATURE_BINNING 和 APPLY_LOG_TRANSFORM 控制) ...
cols_for_binning = ['user_log_count', 'um_log_count', 'merchant_log_count', 'user_time_stamp_nunique']
if APPLY_FEATURE_BINNING:
    print("应用特征分箱...")
    for col in cols_for_binning:
        if col in all_df.columns and all_df[col].notna().any(): # Check if column exists and has non-NaN values
            bin_col_name = f'{col}_binned'
            temp_series = all_df[col].fillna(all_df[col].median())
            if temp_series.nunique() > 1:
                try:
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=min(200_000, len(temp_series)-1 if len(temp_series)>1 else 1), random_state=42)
                    all_df[bin_col_name] = discretizer.fit_transform(temp_series.values.reshape(-1,1)).astype(int)
                    if bin_col_name not in categorical_feats_lgbm: categorical_feats_lgbm.append(bin_col_name)
                except ValueError as e_bin: print(f"分箱列 {col} 失败: {e_bin}. 使用原值。")
            else: all_df[bin_col_name] = temp_series.astype(int) # Or just don't create binned if not enough unique
        else: print(f"列 {col} 不存在或全为NaN，跳过分箱。")


cols_for_log_transform = ['user_log_count', 'um_log_count', 'merchant_log_count', 'user_action_type_0_count', 'user_action_type_1_count', 'user_action_type_2_count', 'user_action_type_3_count']
if APPLY_LOG_TRANSFORM:
    print("应用对数变换...")
    for col in cols_for_log_transform:
        if col in all_df.columns:
            # Ensure non-negative before log1p, fillna(0) helps here if counts can be NaN
            all_df[f'{col}_log1p'] = np.log1p(all_df[col].fillna(0).clip(lower=0)) # clip to ensure non-negative

# 4.B 初步特征过滤 (VarianceThreshold)
# ... (VarianceThreshold逻辑同前一版，由 APPLY_VARIANCE_THRESHOLD 控制) ...
if APPLY_VARIANCE_THRESHOLD:
    print("应用VarianceThreshold...")
    temp_train_df_for_var = all_df[all_df['origin'] == 'train'].copy()
    cols_for_variance = temp_train_df_for_var.columns.drop(['user_id', 'merchant_id', 'label', 'origin'] + categorical_feats_lgbm, errors='ignore')
    numeric_cols_for_variance = temp_train_df_for_var[cols_for_variance].select_dtypes(include=np.number).columns
    if not numeric_cols_for_variance.empty:
        selector_var = VarianceThreshold(threshold=0.01)
        temp_numeric_df = temp_train_df_for_var[numeric_cols_for_variance].fillna(0)
        try:
            selector_var.fit(temp_numeric_df)
            selected_by_var = numeric_cols_for_variance[selector_var.get_support()]
            dropped_by_var = list(set(numeric_cols_for_variance) - set(selected_by_var))
            if dropped_by_var:
                print(f"VarianceThreshold移除了 {len(dropped_by_var)} 个特征.")
                all_df = all_df.drop(columns=dropped_by_var, errors='ignore')
        except ValueError as e_var: print(f"VarianceThreshold执行错误: {e_var}")
        del temp_numeric_df; gc.collect()
    del temp_train_df_for_var; gc.collect()


# --- 5. 模型训练与评估 ---
print("\n--- 5. 模型训练与评估 ---")
# ... (数据准备、SelectKBest, RFE, Sanitizing, Aligning 同前一版，由开关控制) ...
final_train_df = all_df[all_df['origin'] == 'train'].copy()
final_test_df = all_df[all_df['origin'] == 'test'].copy()
del all_df; gc.collect()
final_train_df['label'] = pd.to_numeric(final_train_df['label'], errors='coerce').fillna(0).astype(int)

features_to_drop_model = ['user_id', 'merchant_id', 'label', 'origin', 'time_stamp']
if APPLY_LOG_TRANSFORM: features_to_drop_model.extend([col for col in cols_for_log_transform if f'{col}_log1p' in final_train_df.columns and col in final_train_df.columns]) # Drop original if transformed version exists
if APPLY_FEATURE_BINNING: features_to_drop_model.extend([col for col in cols_for_binning if f'{col}_binned' in final_train_df.columns and col in final_train_df.columns]) # Drop original if binned version exists
features_to_drop_model = list(set([col for col in features_to_drop_model if col in final_train_df.columns]))

X = final_train_df.drop(columns=features_to_drop_model, errors='ignore')
y = final_train_df['label']
X_submission = final_test_df.drop(columns=features_to_drop_model, errors='ignore')

if APPLY_SELECTKBEST:
    print("应用SelectKBest...")
    numeric_cols_kbest = X.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_kbest:
        X_kbest_temp = X[numeric_cols_kbest].replace([np.inf, -np.inf], np.nan).fillna(X[numeric_cols_kbest].median())
        k_val = min(100, X_kbest_temp.shape[1]) 
        if k_val > 0 :
            selector_kbest = SelectKBest(score_func=f_classif, k=k_val)
            try:
                selector_kbest.fit(X_kbest_temp, y)
                selected_by_kbest_numeric = X_kbest_temp.columns[selector_kbest.get_support()].tolist()
                non_numeric_cols_kbest = list(set(X.columns) - set(numeric_cols_kbest))
                final_selected_cols = list(set(selected_by_kbest_numeric + non_numeric_cols_kbest))
                print(f"SelectKBest选择了 {len(selected_by_kbest_numeric)} 数值特征. 总特征数: {len(final_selected_cols)}")
                X = X[final_selected_cols]; X_submission = X_submission[final_selected_cols]
            except Exception as e: print(f"SelectKBest error: {e}")
        else: print("KBest k_val is 0 or less.")
    else: print("No numeric features for SelectKBest.")
    if 'X_kbest_temp' in locals(): del X_kbest_temp; gc.collect()

if APPLY_RFE: 
    print("应用RFE (可能极度耗时)...")
    estimator_rfe = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1)
    n_features_rfe = min(50, X.shape[1]) 
    if n_features_rfe > 0:
        selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=n_features_rfe, step=max(1, int(X.shape[1]*0.1)), verbose=0) # step as int
        X_rfe_temp = X.replace([np.inf, -np.inf], np.nan)
        # Fill NaNs by column median for RFE
        for col in X_rfe_temp.columns:
            if X_rfe_temp[col].isnull().any(): X_rfe_temp[col] = X_rfe_temp[col].fillna(X_rfe_temp[col].median())
        X_rfe_temp = X_rfe_temp.fillna(0) # Catch any remaining NaNs (e.g. if median was NaN)
        
        try:
            selector_rfe.fit(X_rfe_temp, y)
            selected_by_rfe = X.columns[selector_rfe.support_].tolist()
            print(f"RFE选择了 {len(selected_by_rfe)} 个特征.")
            X = X[selected_by_rfe]; X_submission = X_submission[selected_by_rfe]
        except Exception as e: print(f"RFE error: {e}")
        if 'X_rfe_temp' in locals(): del X_rfe_temp; gc.collect()
    else: print("No features for RFE or n_features_to_select is 0.")


X = sanitize_lgbm_cols(X); X_submission = sanitize_lgbm_cols(X_submission)
common_cols = X.columns.intersection(X_submission.columns).tolist()
if not common_cols and (not X.empty and not X_submission.empty) : # Check if common_cols is empty but X and X_sub are not
    # This might happen if sanitization created different col names due to very subtle diffs
    # Or if feature selection was applied only to X.
    # Fallback: try to align based on original order if number of cols match
    print(f"警告: X ({X.shape[1]} cols) 和 X_submission ({X_submission.shape[1]} cols) 清理后无共同列名。将尝试按列顺序对齐。")
    if X.shape[1] == X_submission.shape[1]:
        X_submission.columns = X.columns
        common_cols = X.columns.tolist()
    else:
        raise ValueError("X 和 X_submission 清理后无共同列名且列数不匹配。请检查特征工程和选择步骤。")
elif not common_cols and (X.empty or X_submission.empty):
     raise ValueError("X 或 X_submission 为空。请检查之前的步骤。")

X, X_submission = X[common_cols], X_submission[common_cols]
print(f"最终用于建模的训练特征形状: {X.shape}")


# 5.2 类别不平衡处理 & 5.2.1 超参数调优 & 5.3 模型训练与交叉验证
# ... (逻辑同前一版，使用 IMBALANCE_STRATEGY, DO_HYPERPARAM_TUNING, TUNING_METHOD, DO_POST_CV_FEATURE_SELECTION 控制) ...
# ... (确保所有评估指标 (Precision, Recall, F1) 都被计算和打印) ...
X_train_model, y_train_model = X.copy(), y.copy()
lgb_final_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'learning_rate': 0.01, 'num_leaves': 42,
    'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42
}
# ... (IMBALANCE_STRATEGY application as before) ...
if IMBALANCE_STRATEGY == "smote":
    # ... SMOTE logic ...
    print("处理类别不平衡 (SMOTE)...") 
    original_counts = y_train_model.value_counts()
    k_neighbors_smote = 5
    if not original_counts.empty and original_counts.min() > 0 :
        if original_counts.min() < k_neighbors_smote + 1: k_neighbors_smote = max(1, original_counts.min() - 1)
        if k_neighbors_smote > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_model, y_train_model = smote.fit_resample(X_train_model, y_train_model)
            print(f"SMOTE后训练特征形状: {X_train_model.shape}, 标签分布:\n{pd.Series(y_train_model).value_counts(normalize=True)}")
        else: print("少数类样本过少，SMOTE中止")
    else: print("标签计数异常，SMOTE中止")
elif IMBALANCE_STRATEGY == "random_undersample":
    print("处理类别不平衡 (RandomUnderSampler)...")
    rus = RandomUnderSampler(random_state=42)
    X_train_model, y_train_model = rus.fit_resample(X_train_model, y_train_model)
    print(f"RandomUnderSampler后训练特征形状: {X_train_model.shape}, 标签分布:\n{pd.Series(y_train_model).value_counts(normalize=True)}")
elif IMBALANCE_STRATEGY == "scale_pos_weight":
    counts = np.bincount(y_train_model);
    if len(counts) == 2 and counts[1] > 0: lgb_final_params['scale_pos_weight'] = counts[0] / counts[1]
    print(f"使用 scale_pos_weight: {lgb_final_params.get('scale_pos_weight', 'N/A'):.2f}")


if DO_HYPERPARAM_TUNING:
    # ... (Hyperparameter tuning logic as before) ...
    print(f"\n执行超参数调优 ({TUNING_METHOD})...")
    param_dist = { 
        'n_estimators': [500, 1000, 2000], 'learning_rate': [0.01, 0.02, 0.05],
        'num_leaves': [31, 40, 50], 'colsample_bytree': [0.7, 0.8], 'subsample': [0.7, 0.8]
    }
    param_grid_gs = {'n_estimators': [1000], 'learning_rate': [0.01], 'num_leaves': [31]}
    base_estimator_params = lgb_final_params.copy()
    for k in (param_dist.keys() if TUNING_METHOD == "RandomizedSearch" else param_grid_gs.keys()): base_estimator_params.pop(k, None)
    estimator_for_tuning = lgb.LGBMClassifier(**base_estimator_params)
    if TUNING_METHOD == "RandomizedSearch":
        search_cv = RandomizedSearchCV(estimator=estimator_for_tuning, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=1)
    else: search_cv = GridSearchCV(estimator=estimator_for_tuning, param_grid=param_grid_gs, scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
    
    # Make sure categorical features are passed if X_train_model contains them
    lgbm_cat_feats_for_tuning = [col for col in categorical_feats_lgbm if col in X_train_model.columns]
    # Sanitize cat feature names for LGBM if not already done on X_train_model
    lgbm_cat_feats_for_tuning_sanitized = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in lgbm_cat_feats_for_tuning]


    print(f"开始 {TUNING_METHOD} (可能需要较长时间)...")
    search_cv.fit(X_train_model, y_train_model, categorical_feature=lgbm_cat_feats_for_tuning_sanitized) # Pass sanitized cat features
    print(f"{TUNING_METHOD} 完成."); print("最佳参数: ", search_cv.best_params_); print("最佳AUC: ", search_cv.best_score_)
    lgb_final_params.update(search_cv.best_params_)
else: print("跳过超参数调优。")

print(f"\n最终模型参数: {lgb_final_params}")
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train_model.shape[0])
submission_preds = np.zeros(X_submission.shape[0])
feature_importance_df_cv = pd.DataFrame()
# ... (CV loop as before, ensure 'categorical_feature' is passed to model.fit using sanitized column names) ...
lgbm_cat_feats_final_sanitized = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in categorical_feats_lgbm if x in X_train_model.columns]

cv_start_time = time.time()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_model, y_train_model)):
    print(f"====== Fold {fold_+1} ======")
    X_trn, y_trn = X_train_model.iloc[trn_idx], y_train_model.iloc[trn_idx]
    X_val, y_val = X_train_model.iloc[val_idx], y_train_model.iloc[val_idx]
    model = lgb.LGBMClassifier(**lgb_final_params)
    model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=[col for col in lgbm_cat_feats_final_sanitized if col in X_trn.columns]) # Use sanitized cat feats
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    submission_preds += model.predict_proba(X_submission)[:, 1] / folds.n_splits
    fold_imp_df = pd.DataFrame({"feature": X_trn.columns, "importance": model.feature_importances_, "fold": fold_ + 1})
    feature_importance_df_cv = pd.concat([feature_importance_df_cv, fold_imp_df], axis=0)
print(f"CV训练耗时: {time.time() - cv_start_time:.2f} 秒")

oof_threshold = 0.5 
oof_binary_preds_final = (oof_preds > oof_threshold).astype(int)
metrics_results = {
    "Strategy": f"ObsP_{PROCESS_OBSERVATION_PERIOD}_Filt_{FILTER_SHORT_OBSERVATION_USERS}_Imb_{IMBALANCE_STRATEGY}_Var_{APPLY_VARIANCE_THRESHOLD}_KB_{APPLY_SELECTKBEST}_RFE_{APPLY_RFE}_Tune_{DO_HYPERPARAM_TUNING}_PostCV_{DO_POST_CV_FEATURE_SELECTION}",
    "OOF_AUC": roc_auc_score(y_train_model, oof_preds) if len(np.unique(y_train_model)) > 1 else 0.5, # Handle single class in y_train_model
    "OOF_Precision": precision_score(y_train_model, oof_binary_preds_final, zero_division=0),
    "OOF_Recall": recall_score(y_train_model, oof_binary_preds_final, zero_division=0),
    "OOF_F1": f1_score(y_train_model, oof_binary_preds_final, zero_division=0),
    "OOF_F1_Macro": f1_score(y_train_model, oof_binary_preds_final, average='macro', zero_division=0)
}
print("\n--- 模型评估结果 ---")
for metric, value in metrics_results.items(): print(f"{metric}: {value if isinstance(value, str) else f'{value:.4f}'}")

if DO_POST_CV_FEATURE_SELECTION:
    # ... (Post-CV feature selection and retraining logic from previous version, with similar checks for categorical features) ...
    print("\n执行CV后特征选择与重训练...")
    mean_feature_importance_initial_cv = feature_importance_df_cv.groupby("feature")["importance"].mean().reset_index()
    
    def select_features_by_lgbm_importance(feature_importance_df_mean, X_original_cols, cumulative_threshold=0.99):
        feature_importance_df_mean = feature_importance_df_mean.sort_values(by='importance', ascending=False)
        non_zero_importance_features = feature_importance_df_mean[feature_importance_df_mean['importance'] > 0]
        if non_zero_importance_features.empty: print("警告: 未找到非零重要性特征。返回所有原始特征。"); return X_original_cols.tolist()
        non_zero_importance_features['cumulative_importance'] = non_zero_importance_features['importance'].cumsum() / non_zero_importance_features['importance'].sum()
        selected_features = non_zero_importance_features[non_zero_importance_features['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
        if not selected_features:
            print(f"警告: 累积重要性阈值 {cumulative_threshold} 未选任何特征。用所有非零特征。"); selected_features = non_zero_importance_features['feature'].tolist()
            if not selected_features: print("警告: 仍未选任何特征。返回所有原始特征。"); return X_original_cols.tolist()
        print(f"选择的特征数量: {len(selected_features)}"); return selected_features

    selected_features_after_cv = select_features_by_lgbm_importance(mean_feature_importance_initial_cv, X_train_model.columns, cumulative_threshold=0.99)
    
    if selected_features_after_cv and len(selected_features_after_cv) < X_train_model.shape[1]:
        X_train_sel = X_train_model[selected_features_after_cv]
        X_submission_sel = X_submission[selected_features_after_cv]
        oof_preds_sel = np.zeros(X_train_sel.shape[0]); submission_preds_sel = np.zeros(X_submission_sel.shape[0])
        lgbm_cat_feats_selected_sanitized = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in categorical_feats_lgbm if x in X_train_sel.columns]

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_sel, y_train_model)):
            X_trn_s, y_trn_s = X_train_sel.iloc[trn_idx], y_train_model.iloc[trn_idx]
            X_val_s, y_val_s = X_train_sel.iloc[val_idx], y_train_model.iloc[val_idx]
            model_s = lgb.LGBMClassifier(**lgb_final_params)
            model_s.fit(X_trn_s, y_trn_s, eval_set=[(X_val_s, y_val_s)], eval_metric='auc',
                        callbacks=[lgb.early_stopping(100, verbose=False)],
                        categorical_feature=[col for col in lgbm_cat_feats_selected_sanitized if col in X_trn_s.columns])
            oof_preds_sel[val_idx] = model_s.predict_proba(X_val_s)[:, 1]
            submission_preds_sel += model_s.predict_proba(X_submission_sel)[:, 1] / folds.n_splits
        
        print(f"特征选择后 CV OOF AUC: {roc_auc_score(y_train_model, oof_preds_sel):.4f}")
        submission_preds = submission_preds_sel 
    else: print("CV后特征选择未改变特征集或未执行。")


# --- 6. 结果提交 ---
# ... (Submission logic from previous version) ...
print("\n--- 6. 结果提交 ---")
final_submission_df = pd.DataFrame({'user_id': test_df_orig['user_id'].values, 'merchant_id': test_df_orig['merchant_id'].values, 'prob': submission_preds})
final_submission_df['user_id'] = final_submission_df['user_id'].astype(int)
final_submission_df['merchant_id'] = final_submission_df['merchant_id'].astype(int)
strategy_filename_part = re.sub(r'[^a-zA-Z0-9_]', '', metrics_results["Strategy"]) # Sanitize filename part
output_filename = f'submission_天猫复购预测_{strategy_filename_part[:100]}.csv' # Limit length
final_submission_df.to_csv(output_filename, index=False)
print(f"提交文件 '{output_filename}' 已生成。")

# --- 7. 可视化示例 ---
# ... (Visualization logic from previous version) ...
print("\n--- 7. 可视化 ---")
if not feature_importance_df_cv.empty:
    final_mean_importance_plot = feature_importance_df_cv.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, max(8, len(final_mean_importance_plot.head(30)) * 0.3)))
    sns.barplot(x="importance", y="feature", data=final_mean_importance_plot.head(30), palette="viridis_r")
    plt.title(f"LGBM Feature Importance (Top 30) - Strategy: {metrics_results['Strategy']}")
    plt.tight_layout(); plt.show()

if 'oof_binary_preds_final' in locals():
    cm = confusion_matrix(y_train_model, oof_binary_preds_final)
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Pred NoRep', 'Pred Rep'], yticklabels=['True NoRep', 'True Rep'])
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title("Confusion Matrix (OOF Predictions)"); plt.show()

print("\n--- 大作业代码框架 (再次增强版) 执行完毕 ---")
