#!/usr/bin/env python
# coding: utf-8

# 生成数据

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 设置随机种子
np.random.seed(2025)

# ================== 生成基础数据 ==================
n_users = 5000  # 总用户量
n_teachers = 50  # 教师数量
n_history_terms = 3  # 历史期次数

# 用户属性表
users = pd.DataFrame({
    'user_id': range(1, n_users+1),
    'city': np.random.choice(['一线','新一线','二线','三线','四线','五线'], 
                           size=n_users, p=[0.15,0.25,0.3,0.2,0.08,0.02]),
    'channel': np.random.choice(['广告','自然搜索','社交媒体','转介绍'], 
                              size=n_users, p=[0.4,0.3,0.2,0.1]),
    'experience_teacher_id': np.random.choice(range(1, n_teachers+1), size=n_users)
})

# 教师分组表
teachers = pd.DataFrame({
    'teacher_id': range(1, n_teachers+1),
    'group': np.random.choice(['A','B'], size=n_teachers, p=[0.5,0.5])
})

# ================== 生成历史续报数据 ==================
# 生成历史续报率基准（个人续报率在5%-10%之间波动）
history_renew = pd.DataFrame({
    'user_id': np.repeat(users.user_id, n_history_terms),
    'term': np.tile(range(1, n_history_terms+1), n_users),
    'base_rate': np.random.uniform(0.05, 0.10, size=n_users*n_history_terms)
})
history_renew=history_renew.reset_index(drop=True)
history_renew['renew_status']=np.random.binomial(1,history_renew['base_rate'])
#history_renew.drop(columns='base_rate')

# ================== 生成实验期数据 ==================
# 合并实验分组
exp_data = users.merge(teachers, left_on='experience_teacher_id', right_on='teacher_id')

# 生成实验期续报概率（控制干扰因素）
exp_data['renew_prob'] = np.clip(
    # 实验效应（A组提升3%）
    0.05 + 0.03*(exp_data['group']=='A') +  
    # 城市线级影响（一线+2%，五线-1%）
    np.select(
        [exp_data.city=='一线', exp_data.city=='五线'],
        [0.02, -0.01],
        default=0
    ) + 
    # 渠道影响（转介绍+1%）
    0.01*(exp_data.channel=='转介绍') + 
    # 历史续报率影响（线性叠加）
    0.3*exp_data.merge(
        history_renew.groupby('user_id')['renew_status'].mean().reset_index(name='hist_rate'),
        on='user_id'
    )['hist_rate'],
    0, 1  # 概率截断
)

exp_data['renew_status'] = np.random.binomial(1, exp_data['renew_prob'])

# 生成电话记录（实验组A覆盖率更高）
call_logs = pd.DataFrame({
    'user_id': np.random.choice(exp_data.user_id, 
                               size=int(n_users*0.7)),  # 总体覆盖率70%
    'call_duration': np.abs(np.random.normal(8, 3, int(n_users*0.7))).round(1)
})
call_logs = call_logs.merge(exp_data[['user_id','group']], on='user_id')
call_logs['call_duration'] = call_logs['call_duration'] + 1*(call_logs['group']=='A')  # 实验组时长+1分钟

# ================== 数据验证 ==================
print("实验组续费率:", exp_data[exp_data.group=='A'].renew_status.mean())
print("对照组续费率:", exp_data[exp_data.group=='B'].renew_status.mean())


# 实验效果分析

# In[2]:


df = exp_data.merge(
    history_renew.groupby('user_id')['renew_status'].mean().reset_index(name='history_renew_rate'),
    on='user_id'
)

# 计算电话覆盖率
call_coverage = call_logs.groupby('user_id').size().reset_index(name='called')
df = df.merge(call_coverage, on='user_id', how='left').fillna({'called':0})

# ================== 实验效应评估 ==================
# 方法：双重差分法（DID）
# 构造实验前后数据
did_data = pd.concat([
    history_renew.assign(period='pre', group=history_renew.user_id.map(df.set_index('user_id')['group'])),
    df[['user_id','renew_status','group']].assign(period='post')
])

# DID模型
import statsmodels.formula.api as smf
did_model = smf.ols(
    "renew_status ~ group*period + city + channel + history_renew_rate",
    data=did_data.merge(df[['user_id','city','channel','history_renew_rate']], on='user_id')
).fit()
print(did_model.summary())


# 从差分结果来看，实验组A相比对照组B的净效应为3.6pp(p=0.000)

# In[4]:


# ================== 异质性分析 ==================
# 不同城市线级的处理效应
city_effect = df.groupby(['group','city'])['renew_status'].mean().unstack()
print("不同城市续费率：\n", city_effect)

# 渠道异质性检验
from scipy.stats import chi2_contingency
channel_effect = df.groupby(['group','channel'])['renew_status'].mean().unstack()
print("不同渠道续费率：\n", channel_effect)

# ================== 业务影响分析 ==================
# 电话覆盖与续费关系
call_effect = df.groupby('called')['renew_status'].mean()
print("\n电话覆盖影响：\n", call_effect)

# 通话时长分位数分析
df_call = df.merge(call_logs.groupby('user_id')['call_duration'].mean().reset_index(), how='left')
df_call['call_duration_q'] = pd.qcut(df_call['call_duration'], 4)
print("\n通话时长分位影响：\n", df_call.groupby('call_duration_q')['renew_status'].mean())


# 城市差异：实验组在三线城市提升5.7pp，五线城市仅提升0.2pp
# 渠道差异：实验组在社交媒体提升3.7pp，转介绍仅提升1.5pp
# 电话覆盖：接受电话访问的用户无明显提升，最佳通话时长在8-10分钟
# 改进建议：优先在一线/新一线城市扩大实验组规模；将通话时长控制在8-10分钟区间；增加社交媒体投入；针对低线城市设计差异化的运营策略
