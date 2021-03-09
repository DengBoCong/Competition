import gc
import numpy as np
from netCDF4 import Dataset
import pandas as pd

# ===============SODA_label============
# label_path = './tcdata/enso_round1_train_20210201/SODA_label.nc'
# label_trans_path = './user_data/'
# nc_label = Dataset(label_path, 'r')
# years = np.array(nc_label['year'][:])
# months = np.array(nc_label['month'][:])
# year_month_index = []
# vs = []
# for i, year in enumerate(years):
#     for j, month in enumerate(months):
#         year_month_index.append('year_{}_month_{}'.format(year, month))
#         vs.append(np.array(nc_label['nino'][i, j]))
# df_SODA_label = pd.DataFrame({'year_month': year_month_index})
# df_SODA_label['year_month'] = year_month_index
# df_SODA_label['label'] = vs
# df_SODA_label.to_csv(label_trans_path + 'df_SODA_label.csv', index=None)
# df_SODA_label.head()
#

# ==================SODA_train===================
# SODA_path = './tcdata/enso_round1_train_20210201/SODA_train.nc'
# nc_SODA = Dataset(SODA_path, 'r')
#
#
# def trans_df(df, vals, lats, lons, years, months):
#     """ (100, 36, 24, 72) -- year, month,lat,lon """
#     for j, lat_ in enumerate(lats):
#         for i, lon_ in enumerate(lons):
#             c = 'lat_lon_{}_{}'.format(int(lat_), int(lon_))
#             v = []
#             for y in range(len(years)):
#                 for m in range(len(months)):
#                     v.append(vals[y, m, j, i])
#             df[c] = v
#     return df
#
#
# year_month_index = []
#
# years = np.array(nc_SODA['year'][:])
# months = np.array(nc_SODA['month'][:])
# lats = np.array(nc_SODA['lat'][:])
# lons = np.array(nc_SODA['lon'][:])
#
# for year in years:
#     for month in months:
#         year_month_index.append('year_{}_month_{}'.format(year, month))
#
# df_sst = pd.DataFrame({'year_month': year_month_index})
# df_t300 = pd.DataFrame({'year_month': year_month_index})
# df_ua = pd.DataFrame({'year_month': year_month_index})
# df_va = pd.DataFrame({'year_month': year_month_index})
#
# df_sst = trans_df(df=df_sst, vals=np.array(nc_SODA['sst'][:]), lats=lats, lons=lons, years=years, months=months)
# df_t300 = trans_df(df=df_t300, vals=np.array(nc_SODA['t300'][:]), lats=lats, lons=lons, years=years, months=months)
# df_ua = trans_df(df=df_ua, vals=np.array(nc_SODA['ua'][:]), lats=lats, lons=lons, years=years, months=months)
# df_va = trans_df(df=df_va, vals=np.array(nc_SODA['va'][:]), lats=lats, lons=lons, years=years, months=months)
#
# label_trans_path = './user_data/'
# df_sst.to_csv(label_trans_path + 'df_sst_SODA.csv', index=None)
# df_t300.to_csv(label_trans_path + 'df_t300_SODA.csv', index=None)
# df_ua.to_csv(label_trans_path + 'df_ua_SODA.csv', index=None)
# df_va.to_csv(label_trans_path + 'df_va_SODA.csv', index=None)


# =====================CMIP_label===================
# label_path = './tcdata/enso_round1_train_20210201/CMIP_label.nc'
# label_trans_path = './user_data/'
# nc_label = Dataset(label_path, 'r')
#
# years = np.array(nc_label['year'][:])
# months = np.array(nc_label['month'][:])
#
# year_month_index = []
# vs = []
# for i, year in enumerate(years):
#     for j, month in enumerate(months):
#         year_month_index.append('year_{}_month_{}'.format(year, month))
#         vs.append(np.array(nc_label['nino'][i, j]))
#
# df_CMIP_label = pd.DataFrame({'year_month': year_month_index})
# df_CMIP_label['year_month'] = year_month_index
# df_CMIP_label['label'] = vs
#
# df_CMIP_label.to_csv(label_trans_path + 'df_CMIP_label.csv', index=None)
# df_CMIP_label.head()

# =====================CMIP_train===================
# CMIP_path = './tcdata/enso_round1_train_20210201/CMIP_train.nc'
# CMIP_trans_path = './user_data'
# nc_CMIP = Dataset(CMIP_path, 'r')
#
# year_month_index = []
#
# years = np.array(nc_CMIP['year'][:])
# months = np.array(nc_CMIP['month'][:])
# lats = np.array(nc_CMIP['lat'][:])
# lons = np.array(nc_CMIP['lon'][:])
#
# last_thre_years = 1000
# for year in years:
#     """ 数据的原因，暂时取最后1000个year的数据 """
#     if year >= 4645 - last_thre_years:
#         for month in months:
#             year_month_index.append('year_{}_month_{}'.format(year, month))
#
# df_CMIP_sst = pd.DataFrame({'year_month': year_month_index})
# df_CMIP_t300 = pd.DataFrame({'year_month': year_month_index})
# df_CMIP_ua = pd.DataFrame({'year_month': year_month_index})
# df_CMIP_va = pd.DataFrame({'year_month': year_month_index})
#
#
# def trans_thre_df(df, vals, lats, lons, years, months, last_thre_years=1000):
#     """ (4645, 36, 24, 72) -- year, month,lat,lon """
#     for j, lat_ in (enumerate(lats)):
#         #         print(j)
#         for i, lon_ in enumerate(lons):
#             c = 'lat_lon_{}_{}'.format(int(lat_), int(lon_))
#             v = []
#             for y_, y in enumerate(years):
#                 """ 数据的原因，暂时取最后1000个year的数据 """
#                 if y >= 4645 - last_thre_years:
#                     for m_, m in enumerate(months):
#                         v.append(vals[y_, m_, j, i])
#             df[c] = v
#     return df
#
#
# df_CMIP_sst = trans_thre_df(df=df_CMIP_sst, vals=np.array(nc_CMIP['sst'][:]), lats=lats, lons=lons, years=years,
#                             months=months)
# df_CMIP_sst.to_csv(CMIP_trans_path + 'df_CMIP_sst.csv', index=None)
# del df_CMIP_sst
# gc.collect()
#
# df_CMIP_t300 = trans_thre_df(df=df_CMIP_t300, vals=np.array(nc_CMIP['t300'][:]), lats=lats, lons=lons, years=years,
#                              months=months)
# df_CMIP_t300.to_csv(CMIP_trans_path + 'df_CMIP_t300.csv', index=None)
# del df_CMIP_t300
# gc.collect()
#
# df_CMIP_ua = trans_thre_df(df=df_CMIP_ua, vals=np.array(nc_CMIP['ua'][:]), lats=lats, lons=lons, years=years,
#                            months=months)
# df_CMIP_ua.to_csv(CMIP_trans_path + 'df_CMIP_ua.csv', index=None)
# del df_CMIP_ua
# gc.collect()
#
# df_CMIP_va = trans_thre_df(df=df_CMIP_va, vals=np.array(nc_CMIP['va'][:]), lats=lats, lons=lons, years=years,
#                            months=months)
# df_CMIP_va.to_csv(CMIP_trans_path + 'df_CMIP_va.csv', index=None)
# del df_CMIP_va
# gc.collect()


# ==========================读取训练验证数据=======================
# df_SODA_label = pd.read_csv('./user_data/df_SODA_label.csv')
# # df_CMIP_label = pd.read_csv('./user_data/df_CMIP_label.csv')
#
# df_SODA_label['year'] = df_SODA_label['year_month'].apply(lambda x: x[:x.find('m') - 1])
# df_SODA_label['month'] = df_SODA_label['year_month'].apply(lambda x: x[x.find('m'):])
#
# df_train = pd.pivot_table(data=df_SODA_label, values='label', index='year', columns='month')
# year_new_index = ['year_{}'.format(i + 1) for i in range(df_train.shape[0])]
# month_new_columns = ['month_{}'.format(i + 1) for i in range(df_train.shape[1])]
# df_train = df_train[month_new_columns].loc[year_new_index]
#
# feature_cols = ['month_{}'.format(i + 1) for i in range(12)]
# label_cols = ['month_{}'.format(i + 1) for i in range(12, df_train.shape[1])]
# print(feature_cols)
# print(label_cols)

from code.preprocess import preprocess_data_diff
preprocess_data_diff("./tcdata/enso_round1_train_20210201/SODA_train.nc",
                     "./tcdata/enso_round1_train_20210201/SODA_label.nc",
                     "./user_data/train/{}_{}", "soda")
# preprocess_cmip_data("./tcdata/enso_round1_train_20210201/CMIP_train.nc", "./tcdata/enso_round1_train_20210201/CMIP_label.nc")
# a = np.array([1,2,3])
# b = a[1:2].tolist()
# b[0] = 10
# print(a, b)

# from code.layers import triangular_causal_mask
# print(triangular_causal_mask(30, 10))
# import tensorflow as tf
# import numpy as np
# a = np.arange(1., 6.)
# print(tf.math.top_k(a, 3, sorted=False))