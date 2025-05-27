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


# ++++++++++++++++ ADDED FUNCTION DEFINITION ++++++++++++++++
def sanitize_lgbm_cols(df_or_series_or_list):
    """
    Sanitizes column names for LightGBM compatibility.
    Can handle a DataFrame, a Series (for its name), or a list of column names.
    Replaces non-alphanumeric characters with underscores.
    """
    if isinstance(df_or_series_or_list, pd.DataFrame):
        original_cols = df_or_series_or_list.columns.tolist()
        sanitized_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in original_cols]
        df_or_series_or_list.columns = sanitized_cols
        return df_or_series_or_list
    elif isinstance(df_or_series_or_list, pd.Series):
        original_name = df_or_series_or_list.name
        if original_name:
            sanitized_name = re.sub(r'[^A-Za-z0-9_]+', '_', str(original_name))
            df_or_series_or_list.name = sanitized_name
        return df_or_series_or_list
    elif isinstance(df_or_series_or_list, list):
        return [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_or_series_or_list]
    elif isinstance(df_or_series_or_list, str): # Handle single string if passed
        return re.sub(r'[^A-Za-z0-9_]+', '_', df_or_series_or_list)
    else:
        return df_or_series_or_list
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --- 0. 全局配置与实验开关 ---
print("--- 0. 全局配置与实验开关 ---")
DATA_PATH = './'
# FILE_USER_LOG_PART = DATA_PATH + 'user_log_format1.csv' # Using full log
FILE_USER_LOG_PART = DATA_PATH + 'user_log_format1_part_1.csv' # For faster testing with a smaller part
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
print(f"原始训练集 train_df_orig 中重复行数: {train_df_orig.duplicated().sum()}") #
print(f"原始测试集 test_df_orig 中重复行数: {test_df_orig.duplicated().sum()}") #
# user_info 表中 user_id 应唯一
if 'user_id' in user_info_df.columns: #
    print(f"用户信息表 user_info_df 中 user_id 重复数: {user_info_df.duplicated(subset=['user_id']).sum()}") #
    if user_info_df.duplicated(subset=['user_id']).sum() > 0: #
        user_info_df = user_info_df.drop_duplicates(subset=['user_id'], keep='first') #
        print("已移除 user_info_df 中的重复user_id。") #
else:
    print("警告: user_info_df 中未找到 'user_id' 列。") #


print(f"训练集标签分布:\n{train_df_orig['label'].value_counts(normalize=True)}") #


# --- 2. 数据预处理与清洗 ---
print("\n--- 2. 数据预处理与清洗 ---") #
# 2.1 user_info_df 清洗
user_info_df['age_range'] = user_info_df['age_range'].fillna(0) #
user_info_df['age_range'] = user_info_df['age_range'].replace({7: 6, 8: 6}) # 7,8代表>=50,统一为6 #
user_info_df['gender'] = user_info_df['gender'].fillna(2) # 2代表未知 #

# 2.2 user_log_df 清洗与准备
if 'seller_id' in user_log_df.columns: #
    user_log_df.rename(columns={'seller_id': 'merchant_id'}, inplace=True) #
user_log_df['brand_id'] = user_log_df['brand_id'].fillna(0) # 0代表未知品牌 #

# 时间戳处理，将mmdd格式转换为年内天数 (abs_day) 和月份 (month)
# 数据集时间范围为5月到11月
days_in_month = {5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30} # 11月只到11日，但计算基准按整月 #
month_start_day = {5:1} # 5月1日是第一天 #
for m_idx in range(6, 12): # 6月到11月 #
    month_start_day[m_idx] = month_start_day[m_idx-1] + days_in_month[m_idx-1] #

def mmdd_to_day_number(mmdd): #
    if pd.isna(mmdd) or mmdd == 0: return np.nan #
    # 处理可能存在的浮点数时间戳
    mmdd = int(mmdd) #
    month, day = mmdd // 100, mmdd % 100 #
    
    # 确保月份在定义的字典中
    base_day = month_start_day.get(month) #
    if base_day is None: return np.nan # 无效月份 #
    
    return base_day + day -1 # 例如5月1日是第0天或第1天，根据你的基准 #

user_log_df['abs_day'] = user_log_df['time_stamp'].apply(mmdd_to_day_number) #
user_log_df['month'] = (user_log_df['time_stamp'] // 100).astype(int) # 确保月份是整数 #
# 重要：移除无法计算abs_day的行 (例如，如果time_stamp本身是NaN或无效格式)
user_log_df.dropna(subset=['abs_day'], inplace=True) #
user_log_df['abs_day'] = user_log_df['abs_day'].astype(int) # 确保abs_day是整数 #


def reduce_mem_usage(df, verbose=True): #
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] #
    start_mem = df.memory_usage(deep=True).sum() / 1024**2 #
    for col in df.columns: #
        col_type = df[col].dtypes #
        if col_type in numerics: #
            c_min, c_max = df[col].min(), df[col].max() #
            if pd.isna(c_min) or pd.isna(c_max): continue # Skip if all NaNs or mixed type causing issues #
            if str(col_type)[:3] == 'int': #
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8) #
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16) #
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32) #
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64) #
            else: # float #
                if df[col].apply(lambda x: x.is_integer() if pd.notnull(x) else True).all(): # if all are integers or NaN #
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8) #
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16) #
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32) #
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64) #
                    else: #
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16) #
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32) #
                        else: df[col] = df[col].astype(np.float64) #
                else: # Regular float #
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: df[col] = df[col].astype(np.float16) #
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32) #
                    else: df[col] = df[col].astype(np.float64) #
    end_mem = df.memory_usage(deep=True).sum() / 1024**2 #
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)') #
    return df #

print("优化内存...") #
user_log_df = reduce_mem_usage(user_log_df) #
user_info_df = reduce_mem_usage(user_info_df) #
train_df_orig = reduce_mem_usage(train_df_orig, verbose=False) #
test_df_orig = reduce_mem_usage(test_df_orig, verbose=False) #

# 2.3 合并数据
train_df = pd.merge(train_df_orig, user_info_df, on='user_id', how='left') #
test_df = pd.merge(test_df_orig, user_info_df, on='user_id', how='left') #
train_df['origin'] = 'train'; test_df['origin'] = 'test' #
all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False) #
all_df.drop(columns=['prob'], inplace=True, errors='ignore') # prob列是测试集用于填写的，训练集不需要 #

# --- 3. 特征工程 (实现所有讨论的思路) ---
print("\n--- 3. 特征工程 (完整增强版) ---") #

def get_time_diff_stats(series_abs_day): #
    series_abs_day = pd.to_numeric(series_abs_day, errors='coerce').dropna().sort_values() #
    if len(series_abs_day) < 2: #
        return pd.Series({ #
            'mean_time_diff': np.nan, 'median_time_diff': np.nan, 'std_time_diff': np.nan, #
            'min_time_diff': np.nan, 'max_time_diff': np.nan #
        })
    diffs = series_abs_day.diff().dropna() #
    if diffs.empty: #
        return pd.Series({ #
            'mean_time_diff': np.nan, 'median_time_diff': np.nan, 'std_time_diff': np.nan, #
            'min_time_diff': np.nan, 'max_time_diff': np.nan #
        })
    return pd.Series({ #
        'mean_time_diff': diffs.mean(), 'median_time_diff': diffs.median(), 'std_time_diff': diffs.std(), #
        'min_time_diff': diffs.min(), 'max_time_diff': diffs.max() #
    })

# 3.2 用户整体行为特征
print("衍生用户整体行为特征...") #
user_behavior_features_agg = { #
    'user_log_count': ('item_id', 'count'), 'user_item_nunique': ('item_id', 'nunique'), #
    'user_cat_nunique': ('cat_id', 'nunique'), 'user_merchant_nunique': ('merchant_id', 'nunique'), #
    'user_brand_nunique': ('brand_id', lambda x: x[x != 0].nunique()), # 排除填充的0品牌 #
    'user_time_stamp_nunique': ('time_stamp', 'nunique'), # 原始mmdd格式的天数多样性 #
    'user_abs_day_nunique': ('abs_day', 'nunique'), # 绝对天数的多样性 (更准确的活跃天数) #
    'user_month_nunique': ('month', 'nunique'), #
    'user_action_type_0_count': ('action_type', lambda x: (x==0).sum()), 'user_action_type_1_count': ('action_type', lambda x: (x==1).sum()), #
    'user_action_type_2_count': ('action_type', lambda x: (x==2).sum()), 'user_action_type_3_count': ('action_type', lambda x: (x==3).sum()), #
    'user_first_abs_day': ('abs_day', 'min'), 'user_last_abs_day': ('abs_day', 'max'), #
}
user_behavior_features = user_log_df.groupby('user_id').agg(**user_behavior_features_agg).reset_index() #
user_behavior_features['user_avg_logs_per_active_day'] = user_behavior_features['user_log_count'] / (user_behavior_features['user_abs_day_nunique'] + 1e-6) #
for i in range(4): user_behavior_features[f'user_action_type_{i}_ratio'] = user_behavior_features[f'user_action_type_{i}_count'] / (user_behavior_features['user_log_count'] + 1e-6) #

if ADD_BUY_INTERVAL_FEATURES: #
    print("Calculating user buy interval features...") #
    purchase_logs = user_log_df[user_log_df['action_type'] == 2].sort_values(by=['user_id', 'abs_day']) #
    if not purchase_logs.empty: #
        purchase_logs['buy_interval'] = purchase_logs.groupby('user_id')['abs_day'].diff() #
        user_buy_interval_stats = purchase_logs.groupby('user_id')['buy_interval'].agg( #
            u_mean_buy_interval='mean', u_median_buy_interval='median', u_std_buy_interval='std' #
        ).reset_index() #
        user_behavior_features = pd.merge(user_behavior_features, user_buy_interval_stats, on='user_id', how='left') #
    else: # No purchase logs at all #
        for col in ['u_mean_buy_interval', 'u_median_buy_interval', 'u_std_buy_interval']: user_behavior_features[col] = np.nan #


if ADD_COMPLEX_CONVERSION_RATES: #
    user_behavior_features['user_click_to_addcart_ratio'] = user_behavior_features['user_action_type_1_count'] / (user_behavior_features['user_action_type_0_count'] + 1e-6) #
    user_behavior_features['user_addcart_to_buy_ratio'] = user_behavior_features['user_action_type_2_count'] / (user_behavior_features['user_action_type_1_count'] + 1e-6) #
    user_behavior_features['user_fav_to_buy_ratio'] = user_behavior_features['user_action_type_2_count'] / (user_behavior_features['user_action_type_3_count'] + 1e-6) #

if ADD_ENTROPY_FEATURES: #
    print("Calculating user entropy features...") #
    if not user_log_df.empty: #
        user_cat_entropy = user_log_df.groupby('user_id')['cat_id'].apply(lambda x: entropy(x.value_counts(normalize=True)) if not x.empty and x.nunique() > 1 else np.nan).reset_index().rename(columns={'cat_id':'user_cat_entropy'}) #
        user_behavior_features = pd.merge(user_behavior_features, user_cat_entropy, on='user_id', how='left') #
    else: user_behavior_features['user_cat_entropy'] = np.nan #

print("Calculating user time difference features (may take time)...") #
if not user_log_df.empty: #
    user_time_diffs_series = user_log_df.groupby('user_id')['abs_day'].apply(get_time_diff_stats) #
    if not user_time_diffs_series.empty: #
        if isinstance(user_time_diffs_series.index, pd.MultiIndex): #
            user_time_diffs = user_time_diffs_series.unstack() #
        else: #
             user_time_diffs = pd.DataFrame(user_time_diffs_series.tolist(), index=user_time_diffs_series.index) #
        user_time_diffs = user_time_diffs.reset_index() #
        user_time_diffs = user_time_diffs.rename(columns=lambda c: 'u_' + c if c not in ['user_id'] else c) #
        user_behavior_features = pd.merge(user_behavior_features, user_time_diffs, on='user_id', how='left') #
    else: #
        for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']: #
            user_behavior_features[f'u_{col_suffix}'] = np.nan #
else: # Create empty columns if user_log_df is empty #
    for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']: #
        user_behavior_features[f'u_{col_suffix}'] = np.nan #

all_df = pd.merge(all_df, user_behavior_features, on='user_id', how='left'); gc.collect() #

# 3.3 用户-商家互动特征
print("衍生用户-商家互动特征...") #
um_interaction_agg = { #
    'um_log_count':('item_id','count'), 'um_item_nunique':('item_id','nunique'), #
    'um_cat_nunique':('cat_id','nunique'), 'um_brand_nunique':('brand_id',lambda x: x[x!=0].nunique()), #
    'um_abs_day_nunique':('abs_day','nunique'), # 使用 abs_day 更准确 #
    'um_action_type_0_count':('action_type',lambda x:(x==0).sum()), 'um_action_type_1_count':('action_type',lambda x:(x==1).sum()), #
    'um_action_type_2_count':('action_type',lambda x:(x==2).sum()), 'um_action_type_3_count':('action_type',lambda x:(x==3).sum()), #
    'um_first_abs_day':('abs_day','min'), 'um_last_abs_day':('abs_day','max'), #
}
user_merchant_interaction = user_log_df.groupby(['user_id', 'merchant_id']).agg(**um_interaction_agg).reset_index() #
for i in range(4): user_merchant_interaction[f'um_action_type_{i}_ratio'] = user_merchant_interaction[f'um_action_type_{i}_count'] / (user_merchant_interaction['um_log_count'] + 1e-6) #

print("Calculating user-merchant time difference features (may take time)...") #
if not user_log_df.empty: #
    um_time_diffs_series = user_log_df.groupby(['user_id', 'merchant_id'])['abs_day'].apply(get_time_diff_stats) #
    if not um_time_diffs_series.empty: #
        if isinstance(um_time_diffs_series.index, pd.MultiIndex): #
            um_time_diffs = um_time_diffs_series.unstack() #
        else: #
            um_time_diffs = pd.DataFrame(um_time_diffs_series.tolist(), index=um_time_diffs_series.index) #
        um_time_diffs = um_time_diffs.reset_index() #
        um_time_diffs = um_time_diffs.rename(columns=lambda c: 'um_' + c if c not in ['user_id', 'merchant_id'] else c) #
        user_merchant_interaction = pd.merge(user_merchant_interaction, um_time_diffs, on=['user_id', 'merchant_id'], how='left') #
    else: # Create empty columns #
        for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']: #
            user_merchant_interaction[f'um_{col_suffix}'] = np.nan #
else: # Create empty columns #
    for col_suffix in ['mean_time_diff', 'median_time_diff', 'std_time_diff', 'min_time_diff', 'max_time_diff']: #
        user_merchant_interaction[f'um_{col_suffix}'] = np.nan #


all_df = pd.merge(all_df, user_merchant_interaction, on=['user_id', 'merchant_id'], how='left') #
if 'um_abs_day_nunique' in all_df.columns and 'user_abs_day_nunique' in all_df.columns: #
    all_df['um_active_days_ratio_in_user'] = all_df['um_abs_day_nunique'] / (all_df['user_abs_day_nunique'] + 1e-6) #
else: all_df['um_active_days_ratio_in_user'] = 0 # Or np.nan if preferred #
gc.collect() #

# 3.3.1 时间窗口特征
if ADD_TIME_WINDOW_FEATURES: #
    if 'abs_day' in user_log_df.columns and not user_log_df['abs_day'].empty: # Check if abs_day exists and has valid values #
        max_abs_day_in_log = user_log_df['abs_day'].max() #
        if pd.isna(max_abs_day_in_log): # Handle case where max_abs_day_in_log might be NaN #
             print("无法计算时间窗口特征，日志中的最大有效天数（max_abs_day_in_log）为NaN。") #
        else: #
            for T_window in [7, 15, 30]: #
                print(f"Generating features for last {T_window} days...") #
                window_log_df = user_log_df[user_log_df['abs_day'] > (int(max_abs_day_in_log) - T_window)] #
                
                if not window_log_df.empty: #
                    user_recent_agg_df = window_log_df.groupby('user_id').agg( #
                        **{f'u_logs_last_{T_window}d': ('item_id', 'count'), #
                           f'u_buys_last_{T_window}d': ('action_type', lambda x: (x == 2).sum()), #
                           f'u_cats_last_{T_window}d': ('cat_id', 'nunique'), #
                           f'u_items_last_{T_window}d': ('item_id', 'nunique'), #
                           f'u_active_days_last_{T_window}d': ('abs_day', 'nunique')} # 使用 abs_day #
                    ).reset_index() #
                    all_df = pd.merge(all_df, user_recent_agg_df, on='user_id', how='left') #
                    
                    um_recent_agg_df = window_log_df.groupby(['user_id', 'merchant_id']).agg( #
                        **{f'um_logs_last_{T_window}d': ('item_id', 'count'), #
                           f'um_buys_last_{T_window}d': ('action_type', lambda x: (x == 2).sum())} #
                    ).reset_index() #
                    all_df = pd.merge(all_df, um_recent_agg_df, on=['user_id', 'merchant_id'], how='left') #
                else: # If window_log_df is empty, create placeholder columns with 0 #
                    for prefix_col in ['u', 'um']: #
                        all_df[f'{prefix_col}_logs_last_{T_window}d'] = 0 #
                        all_df[f'{prefix_col}_buys_last_{T_window}d'] = 0 #
                        if prefix_col == 'u': #
                            all_df[f'{prefix_col}_cats_last_{T_window}d'] = 0 #
                            all_df[f'{prefix_col}_items_last_{T_window}d'] = 0 #
                            all_df[f'{prefix_col}_active_days_last_{T_window}d'] = 0 #
                gc.collect() #
    else: print("无法计算时间窗口特征，'abs_day' 列缺失或 user_log_df 为空。") #


# 3.4 商家自身特征
print("衍生商家特征...") #
merchant_features_agg = { #
    'm_user_nunique':('user_id','nunique'), 'm_log_count':('item_id','count'), #
    'm_item_nunique':('item_id','nunique'), 'm_cat_nunique':('cat_id','nunique'), #
    'm_brand_nunique':('brand_id',lambda x:x[x!=0].nunique()), #
    'm_buy_count':('action_type',lambda x:(x==2).sum()), #
    'm_buy_user_nunique':('user_id',lambda x: x[user_log_df.loc[x.index,'action_type']==2].nunique() if not x.empty and not user_log_df.loc[x.index,'action_type'].empty else 0) #
}
merchant_features = user_log_df.groupby('merchant_id').agg(**merchant_features_agg).reset_index() #
merchant_features['m_buyer_conversion_rate'] = merchant_features['m_buy_user_nunique'] / (merchant_features['m_user_nunique'] + 1e-6) #
all_df = pd.merge(all_df, merchant_features, on='merchant_id', how='left'); gc.collect() #

# 3.5 处理特征工程产生的缺失值 (改进版)
print("填充特征工程产生的缺失值...") #
keywords_for_zero_fill = ['count', 'nunique', 'ratio', 'entropy', '_last_', '_buy_', '_logs_', 'active_days'] #_logs_ for _logs_last_Xd #
time_related_cols = [col for col in all_df.columns if 'abs_day' in col or 'time_diff' in col or 'interval' in col] #

for col in all_df.columns: #
    if col in ['user_id', 'merchant_id', 'label', 'origin', 'age_range', 'gender', 'time_stamp']: # Skip IDs, label, origin, and base already handled #
        continue #
    
    if any(keyword in col for keyword in keywords_for_zero_fill): #
        all_df[col] = all_df[col].fillna(0) #
    elif col in time_related_cols: #
        if 'diff' in col or 'interval' in col: # 时间差或间隔，用中位数或0填充 #
            median_val = all_df[col].median() #
            all_df[col] = all_df[col].fillna(median_val if not pd.isna(median_val) else 0) #
        else: # first_abs_day, last_abs_day，用0填充（表示无记录或极早期） #
            all_df[col] = all_df[col].fillna(0) #
    elif pd.api.types.is_numeric_dtype(all_df[col]): # 其他数值型，默认用0填充 #
         all_df[col] = all_df[col].fillna(0) #

all_df['age_range'] = all_df['age_range'].fillna(0) # 0 代表未知 #
all_df['gender'] = all_df['gender'].fillna(2)    # 2 代表未知 #

print("特征工程后 all_df 预览 (部分):") #
all_df.info(verbose=True, show_counts=True); gc.collect() #

# ++++++++++++++++ ADDED CODE: Remove problematic object columns ++++++++++++++++
print("检查并移除潜在的 object 类型 'level_x' 或其他非预期 object 列...") #
cols_to_drop_from_all_df = [] #
# 明确移除从错误信息中得知的列
if 'u_level_1' in all_df.columns and all_df['u_level_1'].dtype == 'object': #
    cols_to_drop_from_all_df.append('u_level_1') #
if 'um_level_2' in all_df.columns and all_df['um_level_2'].dtype == 'object': #
    cols_to_drop_from_all_df.append('um_level_2') #

# 通用检查，移除其他所有 object 类型的列，除了 'origin'
# 因为 'origin' 是我们自己创建的，并且后续会用于分离训练/测试集
# 其他 object 列通常不应该进入模型
for col_name in all_df.columns: #
    if col_name not in ['user_id', 'merchant_id', 'label', 'origin'] and all_df[col_name].dtype == 'object': #
        if col_name not in cols_to_drop_from_all_df: #
             cols_to_drop_from_all_df.append(col_name) #
             print(f"标记移除额外的 object 类型列: {col_name}") #

if cols_to_drop_from_all_df: #
    print(f"将要移除的 object 类型列: {cols_to_drop_from_all_df}") #
    all_df.drop(columns=cols_to_drop_from_all_df, inplace=True, errors='ignore') #
    print(f"已移除 {len(cols_to_drop_from_all_df)} 个 object 类型列。") #
else: #
    print("未发现需要额外移除的 object 类型列。") #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 3.6 处理训练集观察期差异
if PROCESS_OBSERVATION_PERIOD: #
    print("\n--- 3.6 处理训练集观察期差异 ---") #
    if 'user_first_abs_day' in all_df.columns and 'abs_day' in user_log_df.columns and not user_log_df['abs_day'].empty: #
        min_log_day_overall = user_log_df['abs_day'].min() #
        if pd.isna(min_log_day_overall): #
            print("警告: 无法确定日志中的最早有效日期，跳过观察期筛选。") #
        else: #
            grace_period_days = 30 #
            observation_first_day_threshold = min_log_day_overall + grace_period_days #
            
            all_df['user_first_abs_day_numeric'] = pd.to_numeric(all_df['user_first_abs_day'], errors='coerce') #

            condition_short_obs = ( #
                (all_df['origin'] == 'train') &  #
                (all_df['user_first_abs_day_numeric'] > observation_first_day_threshold) &  #
                (all_df['user_first_abs_day_numeric'].notna()) #
            )
            all_df['is_short_observation_train_user'] = 0 #
            all_df.loc[condition_short_obs, 'is_short_observation_train_user'] = 1 #
            print(f"被标记为潜在观察期不足的训练样本数: {all_df[condition_short_obs].shape[0]}") #
            
            if FILTER_SHORT_OBSERVATION_USERS: #
                print("筛选策略：移除观察期不足的训练用户。") #
                original_train_count_before_filter = all_df[all_df['origin'] == 'train'].shape[0] #
                all_df = all_df[~((all_df['origin'] == 'train') & (all_df['is_short_observation_train_user'] == 1))].copy() #
                print(f"筛选前训练样本数: {original_train_count_before_filter}, 筛选后训练集样本数: {all_df[all_df['origin'] == 'train'].shape[0]}") #
                if 'is_short_observation_train_user' in all_df.columns: #
                     all_df.drop(columns=['is_short_observation_train_user'], inplace=True, errors='ignore') #
            
            all_df.drop(columns=['user_first_abs_day_numeric'], inplace=True, errors='ignore') #
    else:  #
        print("警告: 'user_first_abs_day' 特征缺失或 'user_log_df[abs_day]' 为空/无效，无法执行观察期筛选。") #


# --- 4. 特征变换与选择 ---
print("\n--- 4. 特征变换与选择 ---") #
categorical_feats_lgbm = ['age_range', 'gender'] # 基础类别特征 #
if not FILTER_SHORT_OBSERVATION_USERS and PROCESS_OBSERVATION_PERIOD and 'is_short_observation_train_user' in all_df.columns: #
    if 'is_short_observation_train_user' not in categorical_feats_lgbm: #
        categorical_feats_lgbm.append('is_short_observation_train_user') #


# 4.A 数据变换 (分箱, log变换)
cols_for_binning = ['user_log_count', 'um_log_count', 'm_log_count', 'user_abs_day_nunique'] # 使用 'user_abs_day_nunique' 替换 'user_time_stamp_nunique' #
if APPLY_FEATURE_BINNING: #
    print("应用特征分箱...") #
    for col in cols_for_binning: #
        if col in all_df.columns and all_df[col].notna().any(): #
            bin_col_name = f'{col}_binned' #
            temp_series = all_df[col].fillna(all_df[col].median() if not pd.isna(all_df[col].median()) else 0) #
            if temp_series.nunique() > 1: # 确保有足够的多样性去分箱 #
                try: #
                    current_subsample = min(200_000, len(temp_series) -1 if len(temp_series) > 1 else 1) #
                    if current_subsample < 1 : current_subsample = len(temp_series) # 如果计算出的subsample < 1, 使用全部样本 #
                    
                    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile',  #
                                                   subsample=current_subsample, random_state=42) #
                    all_df[bin_col_name] = discretizer.fit_transform(temp_series.values.reshape(-1,1)).astype(int) #
                    if bin_col_name not in categorical_feats_lgbm: categorical_feats_lgbm.append(bin_col_name) #
                except ValueError as e_bin:  #
                    print(f"分箱列 {col} 失败: {e_bin}. 可能由于样本数过少或值分布问题。跳过此列分箱。") #
            else: # 如果唯一值不足，不分箱或直接赋一个默认类别 #
                 print(f"列 {col} 唯一值不足，跳过分箱。") #
        else:  #
            print(f"列 {col} 不存在或全为NaN，跳过分箱。") #

cols_for_log_transform = [ #
    'user_log_count', 'um_log_count', 'm_log_count',  #
    'user_action_type_0_count', 'user_action_type_1_count',  #
    'user_action_type_2_count', 'user_action_type_3_count', #
    'user_item_nunique', 'user_cat_nunique', 'user_merchant_nunique', 'user_brand_nunique', #
    'um_item_nunique', 'um_cat_nunique', 'um_brand_nunique', #
    'm_item_nunique', 'm_cat_nunique', 'm_brand_nunique' #
] 
if APPLY_LOG_TRANSFORM: #
    print("应用对数变换...") #
    for col in cols_for_log_transform: #
        if col in all_df.columns: #
            all_df[f'{col}_log1p'] = np.log1p(all_df[col].clip(lower=0))  #

# 4.B 初步特征过滤 (VarianceThreshold)
if APPLY_VARIANCE_THRESHOLD: #
    print("应用VarianceThreshold...") #
    temp_train_df_for_var = all_df[all_df['origin'] == 'train'].copy() #
    cols_to_exclude_from_variance = ['user_id', 'merchant_id', 'label', 'origin'] + categorical_feats_lgbm #
    numeric_cols_for_variance = temp_train_df_for_var.drop(columns=cols_to_exclude_from_variance, errors='ignore') \
                                                     .select_dtypes(include=np.number).columns #
    
    if not numeric_cols_for_variance.empty: #
        selector_var = VarianceThreshold(threshold=0.01) #
        temp_numeric_df = temp_train_df_for_var[numeric_cols_for_variance].fillna(0) # 用0填充NaN以进行方差计算 #
        try: #
            selector_var.fit(temp_numeric_df) #
            selected_by_var = numeric_cols_for_variance[selector_var.get_support()] #
            dropped_by_var = list(set(numeric_cols_for_variance) - set(selected_by_var)) #
            if dropped_by_var: #
                print(f"VarianceThreshold移除了 {len(dropped_by_var)} 个特征: {dropped_by_var[:10]}...") # 显示部分移除的特征 #
                all_df = all_df.drop(columns=dropped_by_var, errors='ignore') #
            else: print("VarianceThreshold未移除任何特征。") #
        except ValueError as e_var:  #
            print(f"VarianceThreshold执行错误: {e_var}. 可能由于所有特征方差都为0。") #
        del temp_numeric_df; gc.collect() #
    else: print("没有数值型特征可供VarianceThreshold处理。") #
    del temp_train_df_for_var; gc.collect() #


# --- 5. 模型训练与评估 ---
print("\n--- 5. 模型训练与评估 ---") #
final_train_df = all_df[all_df['origin'] == 'train'].copy() #
final_test_df = all_df[all_df['origin'] == 'test'].copy() #
del all_df; gc.collect() #

final_train_df['label'] = pd.to_numeric(final_train_df['label'], errors='coerce').fillna(0).astype(int) #

features_to_drop_model = ['user_id', 'merchant_id', 'label', 'origin', 'time_stamp'] # time_stamp (原始mmdd) 已无用 #
if APPLY_LOG_TRANSFORM:  #
    features_to_drop_model.extend([col for col in cols_for_log_transform if f'{col}_log1p' in final_train_df.columns and col in final_train_df.columns]) #
if APPLY_FEATURE_BINNING:  #
    features_to_drop_model.extend([col for col in cols_for_binning if f'{col}_binned' in final_train_df.columns and col in final_train_df.columns]) #
features_to_drop_model = list(set([col for col in features_to_drop_model if col in final_train_df.columns])) #

X = final_train_df.drop(columns=features_to_drop_model, errors='ignore') #
y = final_train_df['label'] #
X_submission = final_test_df.drop(columns=features_to_drop_model, errors='ignore') #

if APPLY_SELECTKBEST: #
    print("应用SelectKBest...") #
    numeric_cols_kbest = X.select_dtypes(include=np.number).columns.tolist() #
    if numeric_cols_kbest: #
        X_kbest_temp = X[numeric_cols_kbest].replace([np.inf, -np.inf], np.nan) #
        X_kbest_temp = X_kbest_temp.fillna(X_kbest_temp.median().fillna(0)) # 如果中位数也是nan，则用0 #

        k_val = min(100, X_kbest_temp.shape[1])  #
        if k_val > 0 : #
            selector_kbest = SelectKBest(score_func=f_classif, k=k_val) #
            try: #
                selector_kbest.fit(X_kbest_temp, y) #
                selected_by_kbest_numeric = X_kbest_temp.columns[selector_kbest.get_support()].tolist() #
                non_numeric_cols_kbest = list(set(X.columns) - set(numeric_cols_kbest)) #
                final_selected_cols = list(set(selected_by_kbest_numeric + non_numeric_cols_kbest)) #
                
                print(f"SelectKBest选择了 {len(selected_by_kbest_numeric)} 个数值特征. 总特征数变为: {len(final_selected_cols)}") #
                X = X[final_selected_cols]; X_submission = X_submission[final_selected_cols] #
            except Exception as e: print(f"SelectKBest error: {e}") #
        else: print("SelectKBest k_val (要选择的特征数) 为0或更少。") #
        if 'X_kbest_temp' in locals(): del X_kbest_temp; gc.collect() #
    else: print("没有数值型特征可供SelectKBest处理。") #


if APPLY_RFE:  #
    print("应用RFE (可能极度耗时)...") #
    X_rfe_temp = X.replace([np.inf, -np.inf], np.nan) #
    for col in X_rfe_temp.columns: # 逐列填充中位数，然后用0填充剩余NaN #
        if X_rfe_temp[col].isnull().any():  #
            median_val = X_rfe_temp[col].median() #
            X_rfe_temp[col] = X_rfe_temp[col].fillna(median_val if not pd.isna(median_val) else 0) #
    X_rfe_temp = X_rfe_temp.fillna(0) # 再次确保没有NaN #

    estimator_rfe = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1) # n_jobs=-1 for RFE estimator #
    n_features_rfe = min(50, X_rfe_temp.shape[1])  #
    
    if n_features_rfe > 0 and not X_rfe_temp.empty: #
        rfe_step = max(1, int(X_rfe_temp.shape[1] * 0.1)) if X_rfe_temp.shape[1] > 10 else 1 #
        selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=n_features_rfe, step=rfe_step, verbose=0) #
        
        try: #
            selector_rfe.fit(X_rfe_temp, y) #
            selected_by_rfe = X.columns[selector_rfe.support_].tolist() #
            print(f"RFE选择了 {len(selected_by_rfe)} 个特征.") #
            X = X[selected_by_rfe]; X_submission = X_submission[selected_by_rfe] #
        except Exception as e: print(f"RFE error: {e}") #
    else: print("没有特征可供RFE处理或n_features_to_select为0。") #
    if 'X_rfe_temp' in locals(): del X_rfe_temp; gc.collect() #


X = sanitize_lgbm_cols(X.copy()) # 使用 .copy() 避免 SettingWithCopyWarning #
X_submission = sanitize_lgbm_cols(X_submission.copy()) #

common_cols = X.columns.intersection(X_submission.columns).tolist() #
if not common_cols and (not X.empty and not X_submission.empty) : #
    print(f"警告: X ({X.shape[1]} cols) 和 X_submission ({X_submission.shape[1]} cols) 清理后无共同列名。") #
    if X.shape[1] == X_submission.shape[1]: #
        print("列数相同，尝试按列顺序对齐。") #
        X_submission.columns = X.columns # 强制对齐 #
        common_cols = X.columns.tolist() #
    else: #
        x_cols_set = set(X.columns) #
        x_sub_cols_set = set(X_submission.columns) #
        print(f"X独有的列: {x_cols_set - x_sub_cols_set}") #
        print(f"X_submission独有的列: {x_sub_cols_set - x_cols_set}") #
        raise ValueError("X 和 X_submission 清理后无共同列名且列数不匹配。请检查特征工程和选择步骤。") #
elif not common_cols and (X.empty or X_submission.empty): #
     raise ValueError("X 或 X_submission 为空。请检查之前的步骤。") #

if not common_cols and not X.empty and not X_submission.empty: # Should not happen if previous logic is correct #
    print("再次检查：仍然没有共同列，这是一个严重问题。") #
elif common_cols: #
    X = X[common_cols] #
    X_submission = X_submission[common_cols] #
else: # One or both are empty, already raised error #
    pass #


print(f"最终用于建模的训练特征形状: {X.shape}") #
print(f"最终用于建模的测试特征形状: {X_submission.shape}") #


# ++++++++++++++++ 新增代码：输出处理后的数据 ++++++++++++++++
print("\n--- 输出处理后的数据到CSV文件 ---") #
try: #
    processed_train_X_filename = 'processed_train_X.csv' #
    X.to_csv(processed_train_X_filename, index=False) #
    print(f"处理后的训练集特征 X 已保存到: {processed_train_X_filename}") #

    processed_train_y_filename = 'processed_train_y.csv' #
    pd.Series(y, name='label').to_csv(processed_train_y_filename, index=False, header=True) #
    print(f"处理后的训练集标签 y 已保存到: {processed_train_y_filename}") #

    processed_test_X_submission_filename = 'processed_test_X_submission.csv' #
    X_submission.to_csv(processed_test_X_submission_filename, index=False) #
    print(f"处理后的测试集特征 X_submission 已保存到: {processed_test_X_submission_filename}") #

except Exception as e: #
    print(f"输出处理后的数据时发生错误: {e}") #
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 5.2 类别不平衡处理 & 5.2.1 超参数调优 & 5.3 模型训练与交叉验证
X_train_model, y_train_model = X.copy(), y.copy() # 使用X, y的副本进行后续操作 #

lgb_final_params = { #
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', #
    'n_estimators': 3000, 'learning_rate': 0.01, 'num_leaves': 42, # 示例值 #
    'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1, # verbose=-1 抑制LGBM自身输出 #
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1, #
    'random_state': 42 # 对于LGBM来说，seed 和 random_state 效果类似 #
}

if IMBALANCE_STRATEGY == "smote": #
    print("处理类别不平衡 (SMOTE)...")  #
    original_counts = pd.Series(y_train_model).value_counts() #
    k_neighbors_smote = 5 #
    if not original_counts.empty and original_counts.min() > 0 : #
        if original_counts.min() <= k_neighbors_smote : # k_neighbors must be < n_samples in minority class #
             k_neighbors_smote = max(1, original_counts.min() - 1) #
        
        if k_neighbors_smote > 0: # Ensure k_neighbors is still valid #
            try: #
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote) #
                X_train_model, y_train_model = smote.fit_resample(X_train_model, y_train_model) #
                print(f"SMOTE后训练特征形状: {X_train_model.shape}, 标签分布:\n{pd.Series(y_train_model).value_counts(normalize=True)}") #
            except Exception as e_smote: #
                print(f"SMOTE执行错误 (k_neighbors={k_neighbors_smote}): {e_smote}. 可能少数类样本不足。跳过SMOTE。") #
        else:  #
            print("少数类样本过少 (<=1)，SMOTE无法执行。跳过SMOTE。") #
    else:  #
        print("标签计数异常或少数类为0，SMOTE中止。") #

elif IMBALANCE_STRATEGY == "random_undersample": #
    print("处理类别不平衡 (RandomUnderSampler)...") #
    try: #
        rus = RandomUnderSampler(random_state=42) #
        X_train_model, y_train_model = rus.fit_resample(X_train_model, y_train_model) #
        print(f"RandomUnderSampler后训练特征形状: {X_train_model.shape}, 标签分布:\n{pd.Series(y_train_model).value_counts(normalize=True)}") #
    except Exception as e_rus: #
        print(f"RandomUnderSampler执行错误: {e_rus}. 跳过。") #
elif IMBALANCE_STRATEGY == "scale_pos_weight": #
    counts = np.bincount(y_train_model) if isinstance(y_train_model, (np.ndarray, pd.Series)) and y_train_model.ndim == 1 else [] #
    if len(counts) == 2 and counts[1] > 0:  #
        lgb_final_params['scale_pos_weight'] = counts[0] / counts[1] #
        print(f"使用 scale_pos_weight: {lgb_final_params.get('scale_pos_weight', 'N/A'):.2f}") #
    elif len(counts) == 2 and counts[1] == 0: #
        print("警告: 标签中没有正样本，无法计算scale_pos_weight。") #
    else: # 可能标签只有一类，或y_train_model非预期格式 #
        print(f"警告: 无法从标签分布计算scale_pos_weight (标签值计数: {counts})。") #


if DO_HYPERPARAM_TUNING: #
    print(f"\n执行超参数调优 ({TUNING_METHOD})...") #
    param_dist_rs = { # For RandomizedSearch #
        'n_estimators': [500, 1000, 1500, 2000], 'learning_rate': [0.005, 0.01, 0.02, 0.05], #
        'num_leaves': [20, 31, 40, 50, 60], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  #
        'subsample': [0.6, 0.7, 0.8, 0.9], 'reg_alpha': [0, 0.01, 0.1, 0.5], 'reg_lambda': [0, 0.01, 0.1, 0.5] #
    }
    param_grid_gs = { # For GridSearch (smaller grid for speed) #
        'n_estimators': [1000, 1500], 'learning_rate': [0.01, 0.02],  #
        'num_leaves': [31, 42] #
    }
    
    base_estimator_params_for_tuning = lgb_final_params.copy() #
    params_to_tune_in_search = param_dist_rs if TUNING_METHOD == "RandomizedSearch" else param_grid_gs #
    for k in params_to_tune_in_search.keys():  #
        base_estimator_params_for_tuning.pop(k, None) #
    
    estimator_for_tuning = lgb.LGBMClassifier(**base_estimator_params_for_tuning) #
    
    if TUNING_METHOD == "RandomizedSearch": #
        search_cv = RandomizedSearchCV(estimator=estimator_for_tuning, param_distributions=param_dist_rs,  #
                                     n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1, verbose=1) #
    else: # GridSearch #
        search_cv = GridSearchCV(estimator=estimator_for_tuning, param_grid=param_grid_gs,  #
                               scoring='roc_auc', cv=3, n_jobs=-1, verbose=1) #
    
    lgbm_cat_feats_for_tuning = [col for col in categorical_feats_lgbm if col in X_train_model.columns] #
    lgbm_cat_feats_for_tuning_sanitized = sanitize_lgbm_cols(lgbm_cat_feats_for_tuning) #


    print(f"开始 {TUNING_METHOD} (可能需要较长时间)... 使用 {len(lgbm_cat_feats_for_tuning_sanitized)} 个类别特征: {lgbm_cat_feats_for_tuning_sanitized[:5]}...") #
    try: #
        search_cv.fit(X_train_model, y_train_model, categorical_feature=[c for c in lgbm_cat_feats_for_tuning_sanitized if c in X_train_model.columns]) #
        print(f"{TUNING_METHOD} 完成."); print("最佳参数: ", search_cv.best_params_); print("最佳AUC: ", search_cv.best_score_) #
        lgb_final_params.update(search_cv.best_params_) # 更新最终参数 #
    except Exception as e_tune: #
        print(f"超参数调优过程中发生错误: {e_tune}") #
else:  #
    print("跳过超参数调优。") #

print(f"\n最终模型参数: {lgb_final_params}") #
NFOLDS = 5 #
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42) #
oof_preds = np.zeros(X_train_model.shape[0]) # 根据 X_train_model (可能经过SMOTE等) 的形状初始化 #
submission_preds = np.zeros(X_submission.shape[0]) #
feature_importance_df_cv = pd.DataFrame() #

final_sanitized_cat_feats_for_model = sanitize_lgbm_cols( #
    [col for col in categorical_feats_lgbm if col in X_train_model.columns] # 先筛选原始名，再清理 #
)


cv_start_time = time.time() #
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_model, y_train_model)): #
    print(f"====== Fold {fold_+1} ======") #
    X_trn, y_trn_fold = X_train_model.iloc[trn_idx], pd.Series(y_train_model).iloc[trn_idx] # 确保y_train_model是Series #
    X_val, y_val_fold = X_train_model.iloc[val_idx], pd.Series(y_train_model).iloc[val_idx] #
    
    model = lgb.LGBMClassifier(**lgb_final_params) #
    
    current_cat_features_for_fit = [col for col in final_sanitized_cat_feats_for_model if col in X_trn.columns] #
    
    # ++++++++++++++++ DEBUGGING BLOCK START (from previous suggestion, kept for safety) ++++++++++++++++
    # print(f"\n--- Debugging Fold {fold_+1} Data ---")
    # print(f"Shape of X_trn: {X_trn.shape}")
    # print(f"Data types in X_trn before fit:\n{X_trn.dtypes.value_counts()}") 
    # object_cols_in_X_trn = X_trn.select_dtypes(include=['object']).columns
    # if len(object_cols_in_X_trn) > 0:
    #     print(f"!!! CRITICAL ERROR: Found object columns in X_trn for Fold {fold_+1}: {object_cols_in_X_trn.tolist()}")
    #     for obj_col in object_cols_in_X_trn:
    #         print(f"  Unique values in object column '{obj_col}' (first 10): {X_trn[obj_col].unique()[:10]}")
    # else:
    #     print("No object columns found in X_trn. Checking for other potential issues...")
    # print(f"Categorical features passed to fit: {current_cat_features_for_fit}")
    # for cat_f in current_cat_features_for_fit:
    #     if cat_f not in X_trn.columns:
    #         print(f"!!! CRITICAL ERROR: Categorical feature '{cat_f}' is not in X_trn.columns!")
    # print("--- End Debugging Fold Data ---\n")
    # ++++++++++++++++ DEBUGGING BLOCK END ++++++++++++++++
    
    model.fit(X_trn, y_trn_fold, eval_set=[(X_val, y_val_fold)], eval_metric='auc', #
              callbacks=[lgb.early_stopping(100, verbose=False)], # verbose=False 减少输出 #
              categorical_feature=current_cat_features_for_fit)  #
              
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1] #
    if not X_submission.empty: # 确保 X_submission 不是空的 #
        submission_preds += model.predict_proba(X_submission)[:, 1] / folds.n_splits #
        
    fold_imp_df = pd.DataFrame({"feature": X_trn.columns.tolist(), "importance": model.feature_importances_, "fold": fold_ + 1}) #
    feature_importance_df_cv = pd.concat([feature_importance_df_cv, fold_imp_df], axis=0) #
print(f"CV训练耗时: {time.time() - cv_start_time:.2f} 秒") #

oof_threshold = 0.5  #
oof_binary_preds_final = (oof_preds > oof_threshold).astype(int) #

y_true_for_metrics = pd.Series(y_train_model) # 确保是 Series #
if len(y_true_for_metrics) != len(oof_preds): #
    print(f"警告: y_true_for_metrics (len {len(y_true_for_metrics)}) 和 oof_preds (len {len(oof_preds)}) 长度不匹配！评估可能不准确。") #

metrics_results = { #
    "Strategy": f"ObsP{int(PROCESS_OBSERVATION_PERIOD)}_Filt{int(FILTER_SHORT_OBSERVATION_USERS)}_Imb{IMBALANCE_STRATEGY[:3]}_Var{int(APPLY_VARIANCE_THRESHOLD)}_KB{int(APPLY_SELECTKBEST)}_RFE{int(APPLY_RFE)}_Tune{int(DO_HYPERPARAM_TUNING)}_PostCV{int(DO_POST_CV_FEATURE_SELECTION)}", #
    "OOF_AUC": roc_auc_score(y_true_for_metrics, oof_preds) if len(np.unique(y_true_for_metrics)) > 1 and len(y_true_for_metrics) == len(oof_preds) else 0.5, #
    "OOF_Precision": precision_score(y_true_for_metrics, oof_binary_preds_final, zero_division=0) if len(y_true_for_metrics) == len(oof_binary_preds_final) else 0, #
    "OOF_Recall": recall_score(y_true_for_metrics, oof_binary_preds_final, zero_division=0) if len(y_true_for_metrics) == len(oof_binary_preds_final) else 0, #
    "OOF_F1": f1_score(y_true_for_metrics, oof_binary_preds_final, zero_division=0) if len(y_true_for_metrics) == len(oof_binary_preds_final) else 0, #
    "OOF_F1_Macro": f1_score(y_true_for_metrics, oof_binary_preds_final, average='macro', zero_division=0) if len(y_true_for_metrics) == len(oof_binary_preds_final) else 0 #
}
print("\n--- 模型评估结果 ---") #
for metric, value in metrics_results.items(): print(f"{metric}: {value if isinstance(value, str) else f'{value:.4f}'}") #

if DO_POST_CV_FEATURE_SELECTION: #
    print("\n执行CV后特征选择与重训练...") #
    mean_feature_importance_initial_cv = feature_importance_df_cv.groupby("feature")["importance"].mean().reset_index() #
    
    def select_features_by_lgbm_importance(feature_importance_df_mean, X_original_cols, cumulative_threshold=0.99): #
        feature_importance_df_mean = feature_importance_df_mean.sort_values(by='importance', ascending=False) #
        non_zero_importance_features = feature_importance_df_mean[feature_importance_df_mean['importance'] > 0] #
        if non_zero_importance_features.empty:  #
            print("警告: CV后未找到非零重要性特征。返回所有原始特征。") #
            return X_original_cols.tolist() # 返回原始列名 #
        
        non_zero_importance_features['cumulative_importance'] = non_zero_importance_features['importance'].cumsum() / non_zero_importance_features['importance'].sum() #
        selected_features = non_zero_importance_features[non_zero_importance_features['cumulative_importance'] <= cumulative_threshold]['feature'].tolist() #
        
        if not selected_features: # 如果阈值过高导致没有选出特征 #
            print(f"警告: 累积重要性阈值 {cumulative_threshold} 未选任何特征。将使用所有非零重要性特征。") #
            selected_features = non_zero_importance_features['feature'].tolist() #
            if not selected_features: # 仍然没有特征（例如所有特征重要性都是0） #
                print("警告: 仍未选任何特征。返回所有原始特征。") #
                return X_original_cols.tolist() #
        print(f"CV后选择的特征数量: {len(selected_features)}") #
        return selected_features #

    selected_features_after_cv = select_features_by_lgbm_importance(mean_feature_importance_initial_cv, X_train_model.columns, cumulative_threshold=0.99) #
    
    if selected_features_after_cv and len(selected_features_after_cv) < X_train_model.shape[1] and X_train_model.shape[1] > 0 : #
        X_train_sel = X_train_model[selected_features_after_cv] #
        X_submission_sel = X_submission[selected_features_after_cv] # 假设 X_submission 包含这些列 #
        
        oof_preds_sel = np.zeros(X_train_sel.shape[0]) #
        submission_preds_sel = np.zeros(X_submission_sel.shape[0]) if not X_submission_sel.empty else np.array([]) #
        
        lgbm_cat_feats_for_retrain_sanitized = [col for col in final_sanitized_cat_feats_for_model if col in X_train_sel.columns] #

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_sel, y_train_model)): # y_train_model 是SMOTE等处理后的 #
            X_trn_s, y_trn_s_fold = X_train_sel.iloc[trn_idx], pd.Series(y_train_model).iloc[trn_idx] #
            X_val_s, y_val_s_fold = X_train_sel.iloc[val_idx], pd.Series(y_train_model).iloc[val_idx] #
            
            model_s = lgb.LGBMClassifier(**lgb_final_params) #
            current_cat_features_for_retrain_fit = [col for col in lgbm_cat_feats_for_retrain_sanitized if col in X_trn_s.columns] #

            model_s.fit(X_trn_s, y_trn_s_fold, eval_set=[(X_val_s, y_val_s_fold)], eval_metric='auc', #
                        callbacks=[lgb.early_stopping(100, verbose=False)], #
                        categorical_feature=current_cat_features_for_retrain_fit) #
            oof_preds_sel[val_idx] = model_s.predict_proba(X_val_s)[:, 1] #
            if not X_submission_sel.empty: #
                submission_preds_sel += model_s.predict_proba(X_submission_sel)[:, 1] / folds.n_splits #
        
        y_true_for_sel_metrics = pd.Series(y_train_model) # 确保是 Series #
        if len(y_true_for_sel_metrics) == len(oof_preds_sel): #
            print(f"特征选择后 CV OOF AUC: {roc_auc_score(y_true_for_sel_metrics, oof_preds_sel):.4f}") #
        else: #
            print(f"警告: y_true_for_sel_metrics (len {len(y_true_for_sel_metrics)}) 和 oof_preds_sel (len {len(oof_preds_sel)}) 长度不匹配！评估可能不准确。") #

        submission_preds = submission_preds_sel # 更新提交预测 #
    else:  #
        print("CV后特征选择未改变特征集、未执行或X_train_model为空。") #


# --- 6. 结果提交 ---
print("\n--- 6. 结果提交 ---") #
if not X_submission.empty: # 只有当测试集特征存在时才生成提交文件 #
    final_submission_df = pd.DataFrame({'user_id': test_df_orig['user_id'].values,  #
                                      'merchant_id': test_df_orig['merchant_id'].values,  #
                                      'prob': submission_preds if len(submission_preds) == len(test_df_orig) else np.full(len(test_df_orig), 0.5) }) # Fallback if length mismatch #
    final_submission_df['user_id'] = final_submission_df['user_id'].astype(int) #
    final_submission_df['merchant_id'] = final_submission_df['merchant_id'].astype(int) #
    
    strategy_cleaned_for_filename = re.sub(r'[^\w-]', '_', metrics_results["Strategy"]) # 替换非法字符为下划线 #
    output_filename = f'submission_天猫复购预测_{strategy_cleaned_for_filename[:80]}.csv' # 限制文件名长度 #
    
    final_submission_df.to_csv(output_filename, index=False) #
    print(f"提交文件 '{output_filename}' 已生成。") #
else: #
    print("测试集特征 X_submission 为空，不生成提交文件。") #

# --- 7. 可视化示例 ---
print("\n--- 7. 可视化 ---") #
if not feature_importance_df_cv.empty: #
    final_mean_importance_plot = feature_importance_df_cv.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index() #
    plt.figure(figsize=(12, max(8, int(len(final_mean_importance_plot.head(30)) * 0.4)))) # 调整高度计算 #
    sns.barplot(x="importance", y="feature", data=final_mean_importance_plot.head(30), palette="viridis_r") #
    
    title_strategy_part = re.sub(r'[^\w\s-]', '_', metrics_results['Strategy']) #
    plt.title(f"LGBM Feature Importance (Top 30)\nStrategy: {title_strategy_part}", fontsize=10) #
    plt.xticks(fontsize=8) #
    plt.yticks(fontsize=8) #
    plt.tight_layout();  #
    try: #
        plt.show() #
    except Exception as e_plt: print(f"显示特征重要性图表时出错: {e_plt}") #


if 'oof_binary_preds_final' in locals() and 'y_train_model' in locals() and len(y_train_model) == len(oof_binary_preds_final): #
    try: #
        cm = confusion_matrix(y_train_model, oof_binary_preds_final) #
        plt.figure(figsize=(6,5));  #
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  #
                    xticklabels=['Pred NoRep', 'Pred Rep'], yticklabels=['True NoRep', 'True Rep']) #
        plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title("Confusion Matrix (OOF Predictions)") #
        plt.tight_layout();  #
        plt.show() #
    except Exception as e_cm_plt: print(f"显示混淆矩阵图表时出错: {e_cm_plt}") #
else: #
    print("无法生成混淆矩阵：OOF预测或真实标签数据不完整或长度不匹配。") #

print("\n--- 大作业代码框架 (再次增强版) 执行完毕 ---") #
