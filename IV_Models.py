import csv
import numpy as np
import pandas as pd
from statsmodels.api import add_constant
from linearmodels.iv import IV2SLS
from linearmodels.iv import compare

which_site = 'English'

df_covariates = pd.read_csv('Data/'+which_site+ '_Covariates.csv',
                            names = ['site', 'post_id', 'parent_id',
                                     'user_id', 'answer_score', 'all_answer_score',
                                     'answer_score_5p', 'answer_score_10p', 'answer_score_15p',
                                     'answer_score_20p', 'answer_score_25p', 'answer_score_30p',
                                     'answer_score_bar_5p', 'answer_score_bar_10p', 'answer_score_bar_15p',
                                     'answer_score_bar_20p', 'answer_score_bar_25p', 'answer_score_bar_30p',
                                     'answer_position_5p', 'answer_position_10p', 'answer_position_15p',
                                     'answer_position_20p', 'answer_position_25p', 'answer_position_30p',
                                     'answer_comment_count', 'answer_day_of_week', 'answer_time_of_day',
                                     'answer_epoch', 'answer_timeliness', 'answer_order',
                                     'question_score', 'question_view_count', 'question_favorite_count',
                                     'question_comment_count', 'question_answer_count', 'answerer_post_count',
                                     'answerer_answer_count', 'answerer_active_age', 'answerer_proxy_reputation',
                                     'answerer_proxy_reputation_answer', 'answerer_gold_count', 'answerer_silver_count',
                                     'answerer_bronze_count', 'answerer_question_score_total', 'answerer_question_view_total',
                                     'answerer_question_favorite_total', 'answerer_question_comment_total', 'answerer_question_answer_total'])

# Perform log modulo transformation of variables
all_lm_var = ['answer_score', 'answer_score_5p', 'answer_score_10p',
              'answer_score_15p', 'answer_score_20p', 'answer_score_25p',
              'answer_score_30p', 'answer_score_bar_5p', 'answer_score_bar_10p',
              'answer_score_bar_15p', 'answer_score_bar_20p', 'answer_score_bar_25p',
              'answer_score_bar_30p', 'answer_position_5p', 'answer_position_10p',
              'answer_position_15p', 'answer_position_20p', 'answer_position_25p',
              'answer_position_30p', 'answer_comment_count', 'answer_day_of_week',
              'answer_time_of_day', 'answer_epoch', 'answer_timeliness',
              'answer_order', 'question_score', 'question_view_count',
              'question_favorite_count', 'question_comment_count', 'question_answer_count',
              'answerer_post_count', 'answerer_answer_count', 'answerer_active_age',
              'answerer_proxy_reputation', 'answerer_proxy_reputation_answer', 'answerer_gold_count',
              'answerer_silver_count', 'answerer_bronze_count', 'answerer_question_score_total',
              'answerer_question_view_total', 'answerer_question_favorite_total', 'answerer_question_comment_total',
              'answerer_question_answer_total']

for var_lm in all_lm_var:
    df_covariates['lm_'+var_lm] = np.multiply(np.sign(df_covariates[var_lm]), np.log(np.absolute(df_covariates[var_lm])+1))

# Add constant as part of the data
df_covariates = add_constant(df_covariates, has_constant='add')


# We consider two variants of IV model: with and without log modulo transformation
reputation_model = [['lm_answer_score'],
    ['lm_question_score', 'lm_question_view_count', 'lm_question_favorite_count', 'lm_question_comment_count', 'lm_question_answer_count'],
    ['lm_answerer_proxy_reputation', 'lm_answerer_proxy_reputation_answer', 'lm_answerer_gold_count', 'lm_answerer_silver_count', 'lm_answerer_bronze_count'],
    ['lm_answerer_question_score_total', 'lm_answerer_question_view_total', 'lm_answerer_question_favorite_total', 'lm_answerer_question_comment_total', 'lm_answerer_question_answer_total']]
'''
reputation_model = [['answer_score'],
    ['question_score', 'question_view_count', 'question_favorite_count', 'question_comment_count', 'question_answer_count'],
    ['answerer_proxy_reputation', 'answerer_proxy_reputation_answer', 'answerer_gold_count', 'answerer_silver_count', 'answerer_bronze_count'],
    ['answerer_question_score_total', 'answerer_question_view_total', 'answerer_question_favorite_total', 'answerer_question_comment_total', 'answerer_question_answer_total']]
'''

var_dependent = reputation_model[0][0]
all_exogenous = reputation_model[1]
all_endogeneous = reputation_model[2]
all_instrumental = reputation_model[3]

all_results = np.zeros((len(all_instrumental), len(all_endogeneous), 2))

for i, var_endogeneous in enumerate(all_endogeneous):

    print('*******'+var_endogeneous.upper()+'*******', file=open('Results/'+which_site+'_Results.txt', 'a'))
    
    print('OLS model with no control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    res_ols = IV2SLS(df_covariates[var_dependent], df_covariates[[var_endogeneous, 'const']], None, None).fit(cov_type='unadjusted')
    print(res_ols, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

    for j, var_instrumental in enumerate(all_instrumental):
        print('***'+var_endogeneous.upper()+': '+var_instrumental+'***', file=open('Results/'+which_site+'_Results.txt', 'a'))
        
        print('2SLS model with no control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
        # 2SLS function call: IV2SLS(dependent, exogeneous, endogeneous, instrumental)
        res_2sls = IV2SLS(df_covariates[var_dependent], df_covariates['const'], df_covariates[var_endogeneous], df_covariates[var_instrumental]).fit(cov_type='unadjusted')
        print(res_2sls, file=open('Results/'+which_site+'_Results.txt', 'a'))
        print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

        print('OLS model with control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
        var_exogenous = all_exogenous[j]
        all_regressors = [var_exogenous, var_endogeneous, 'const']
        res_ols_ctrl = IV2SLS(df_covariates[var_dependent], df_covariates[all_regressors], None, None).fit(cov_type='unadjusted')
        print(res_ols_ctrl, file=open('Results/'+which_site+'_Results.txt', 'a'))
        print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

        print('2SLS model with control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
        var_exogenous = all_exogenous[j]
        all_control = [var_exogenous, 'const']
        # 2SLS function call: IV2SLS(dependent, exogeneous, endogeneous, instrumental)
        res_2sls_ctrl = IV2SLS(df_covariates[var_dependent], df_covariates[all_control], df_covariates[var_endogeneous], df_covariates[var_instrumental]).fit(cov_type='unadjusted')
        print(res_2sls_ctrl, file=open('Results/'+which_site+'_Results.txt', 'a'))
        print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

    all_exogenous_and_const = all_exogenous[:]
    all_exogenous_and_const.append('const')

    print('***'+var_endogeneous.upper()+': All***', file=open('Results/'+which_site+'_Results.txt', 'a'))
    
    print('2SLS model with no control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    res_2sls_master = IV2SLS(df_covariates[var_dependent], df_covariates['const'], df_covariates[var_endogeneous], df_covariates[all_instrumental]).fit(cov_type='unadjusted')
    print(res_2sls_master, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

    print('OLS model with control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    all_regressors= [elem for elem in all_exogenous]
    all_regressors.extend([var_endogeneous, 'const'])
    res_ols_ctrl_master = IV2SLS(df_covariates[var_dependent], df_covariates[all_regressors], None, None).fit(cov_type='unadjusted')
    print(res_ols_ctrl_master, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

    print('2SLS model with control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    all_control  = [elem for elem in all_exogenous]
    all_control.append('const')
    res_2sls_ctrl_master = IV2SLS(df_covariates[var_dependent], df_covariates[all_control], df_covariates[var_endogeneous], df_covariates[all_instrumental]).fit(cov_type='unadjusted')
    print(res_2sls_ctrl_master, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    
# We consider two variants of IV model: with and without log modulo transformation
'''
herd_model = [['lm_answer_score_bar_5p', 'lm_answer_score_bar_10p', 'lm_answer_score_bar_15p', 'lm_answer_score_bar_20p', 'lm_answer_score_bar_25p', 'lm_answer_score_bar_30p'],
    ['lm_answer_position_5p', 'lm_answer_position_10p', 'lm_answer_position_15p', 'lm_answer_position_20p', 'lm_answer_position_25p', 'lm_answer_position_30p'],
    ['lm_answer_score_5p', 'lm_answer_score_10p', 'lm_answer_score_15p', 'lm_answer_score_20p', 'lm_answer_score_25p', 'lm_answer_score_30p'],
    ['lm_answer_timeliness', 'lm_answer_order']]
'''

herd_model = [['answer_score_bar_5p', 'answer_score_bar_10p', 'answer_score_bar_15p', 'answer_score_bar_20p', 'answer_score_bar_25p', 'answer_score_bar_30p'],
    ['answer_position_5p', 'answer_position_10p', 'answer_position_15p', 'answer_position_20p', 'answer_position_25p', 'answer_position_30p'],
    ['answer_score_5p', 'answer_score_10p', 'answer_score_15p', 'answer_score_20p', 'answer_score_25p', 'answer_score_30p'],
    ['answer_timeliness', 'answer_order']]

all_dependent = herd_model[0]
all_endogeneous_G1 = herd_model[1]
all_endogeneous_G2 = herd_model[2]
all_instrumental = herd_model[3]

for i, var_dependent in enumerate(all_dependent):

    var_endogeneous_G1 = all_endogeneous_G1 [i]
    var_endogeneous_G2 = all_endogeneous_G2 [i]

    print('*******'+var_dependent.upper()+'*******', file=open('Results/'+which_site+'_Results.txt', 'a'))
    
    print('OLS model with no control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    res_ols = IV2SLS(df_covariates[var_dependent], df_covariates[[var_endogeneous_G1, var_endogeneous_G2 ,  'answerer_proxy_reputation_answer', 'const']], None, None).fit(cov_type='unadjusted')
    print(res_ols, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))

    print('2SLS model with no control\n', file=open('Results/'+which_site+'_Results.txt', 'a'))
    # 2SLS function call: IV2SLS(dependent, exogeneous, endogeneous, instrumental)
    res_2sls = IV2SLS(df_covariates[var_dependent], df_covariates[['answerer_proxy_reputation_answer', 'const']], df_covariates[[var_endogeneous_G1, var_endogeneous_G2]], df_covariates[all_instrumental]).fit(cov_type='unadjusted')
    print(res_2sls, file=open('Results/'+which_site+'_Results.txt', 'a'))
    print('*******************************************************************************\n', file=open('Results/'+which_site+'_Results.txt', 'a'))


