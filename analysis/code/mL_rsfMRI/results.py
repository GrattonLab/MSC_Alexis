import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
#import other python scripts for further anlaysis
import reshape
import plotFW
import results
# Initialization of directory information:
thisDir = os.path.expanduser('~/Desktop/MSC_Alexis/analysis/')
dataDir = thisDir + 'data/mvpa_data/'
outDir = thisDir + 'output/mL/'

#Subjects and tasks
taskList=['mixed', 'motor','mem']
#taskList=['pres1', 'pres2','pres3']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
def boxACC(df, classifier, analysis):
    if analysis=='CV':
        print('cross validation boxplots')
        df= pd.melt(df, value_vars=['mixed','motor','mem'], var_name='task', value_name='acc')
        #df.drop('sub', axis=1, inplace=True)
        plt.figure(figsize=(15,8))
        ax=sns.boxplot(x='task', y='acc', data=df)
        ax.set_title('Cross Validation')
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        #fig=ax.get_figure()
        plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')
    elif analysis=='SS':
        print('same sub boxplots')
        #df.drop(['sub'], axis=1, inplace=True)
        plt.figure(figsize=(8,6))
        sns.set_context("talk")
        ax=sns.boxplot(x='test_task', y='acc', hue='train_task', data=df)
        ax.axhline(.50, ls='--', color='r')
        ax.set_title(classifier)
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Train Task',loc='lower left')
        #fig=ax.get_figure()
        plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')
    elif analysis=='DS':
        #df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        plt.figure(figsize=(8,6))
        sns.set_context("talk")
        ax=sns.boxplot(x='task', y='acc', data=df)
        ax.axhline(.50, ls='--', color='r')
        ax.set_title(classifier)
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        #fig=ax.get_figure()
        plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')
    elif analysis=='BS':
        print('diff sub diff task boxplots')
        #df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        plt.figure(figsize=(8,6))
        sns.set_context("talk")
        ax=sns.boxplot(x='test_task', y='acc', hue='train_task', data=df)
        ax.axhline(.50, ls='--', color='r')
        ax.set(ylim=(.4))
        ax.set_title(classifier)
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Train Task',loc='upper right')
        #fig=ax.get_figure()
        plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')
    else:
        print('skipping boxplots')
def plotACC(df, classifier, analysis):
    if analysis == 'CV':
        print('task by subject heatmap')
        plt.figure()
        ax=sns.heatmap(df, annot=True, vmin=.5, vmax=1)
        #fig=ax.get_figure()
        plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/heatmap.png', bbox_inches='tight')
    elif analysis=='SS':
        print('task by subject heatmap')
        grouped_df=df.groupby('test_task')
        for task in taskList:
            task_df=grouped_df.get_group(task)
            task_df.drop(columns=['test_task'], inplace=True)
            task_df=task_df.pivot(index='sub', columns='train_task', values='acc')
            plt.figure()
            ax=sns.heatmap(task_df, annot=True, vmin=.5, vmax=1)
            ax.set_title('Testing ' + task)
            ax.set_xlabel('Training Variables')
            #fig=ax.get_figure()
            plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/'+task+'_heatmap.png', bbox_inches='tight')
    elif analysis =='DS':
        print('subject by subject heatmap')
        grouped_df=df.groupby('task')
        for task in taskList:
            task_df=grouped_df.get_group(task)
            task_df.drop(columns=['task'], inplace=True)
            task_df=task_df.pivot(index='test_sub', columns='train_sub', values='acc')
            plt.figure()
            ax=sns.heatmap(task_df, annot=True, vmin=.5, vmax=1)
            ax.set_title('Testing ' + task)
            ax.set_xlabel('Training Variables')
            ax.set_ylabel('Testing Variables')
            #fig=ax.get_figure()
            plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/'+task+'_heatmap.png', bbox_inches='tight')
    elif analysis =='BS':
        print('subject by subject per train task split heatmap')
        test_group=df.groupby('test_task')
        for test_task in taskList:
            task_df=test_group.get_group(test_task)
            for train_task in taskList:
                if test_task==train_task:
                    continue
                else:
                    df=task_df[task_df['train_task']==train_task]
                    df.drop(columns=['train_task'], inplace=True)
                    df.drop(columns=['test_task'], inplace=True)
                    df=df.pivot(index='test_sub', columns='train_sub', values='acc')
                    plt.figure()
                    ax=sns.heatmap(df, annot=True, vmin=.5, vmax=1)
                    ax.set_title('Testing ' + test_task+' Training '+train_task)
                    ax.set_xlabel('Training Variables')
                    ax.set_ylabel('Testing Variables')
                    #fig=ax.get_figure()
                    plt.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/train_'+train_task+'_test_'+test_task+'_heatmap.png', bbox_inches='tight')
    else:
        print('skipping heatmaps')
def statsACC(df, classifier, analysis):
    if analysis=='CV':
        print('cross validation stats')
        mu=df.mean()
        sd=df.std()
        stats=pd.DataFrame({'Mean':mu, 'Std':sd})
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='SS':
        print('same sub stats')
        df.drop(['sub'], axis=1, inplace=True)
        stats=df.groupby(['test_task', 'train_task']).mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby(['test_task', 'train_task']).std()
        stats['Std']=sd['acc']
        stats.reset_index(inplace=True)
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='DS':
        print('diff sub stats')
        df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        stats=df.groupby('task').mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby('task').std()
        stats['Std']=sd['acc']
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='BS':
        print('diff sub diff task stats')
        df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        stats=df.groupby(['test_task', 'train_task']).mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby(['test_task', 'train_task']).std()
        stats['Std']=sd['acc']
        stats.reset_index(inplace=True)
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    else:
        print('skipping stats')

def cv_modelComp():
    SVC=pd.read_csv(outDir+'results/SVC/acc/CV/acc.csv')
    log=pd.read_csv(outDir+'results/logReg/acc/CV/acc.csv')
    ridge=pd.read_csv(outDir+'results/Ridge/acc/CV/acc.csv')
    SVC.drop(columns='sub', inplace=True)
    SVC_un=SVC.melt(value_vars=['mixed', 'motor','mem'], var_name='Task', value_name='Accuracy')
    SVC_un['Analysis']='SVC'

    log.drop(columns='sub', inplace=True)
    log_un=log.melt(value_vars=['mixed', 'motor','mem'],var_name='Task', value_name='Accuracy')
    log_un['Analysis']='logReg'

    ridge.drop(columns='sub', inplace=True)
    ridge_un=ridge.melt(value_vars=['mixed', 'motor','mem'],var_name='Task', value_name='Accuracy')
    ridge_un['Analysis']='Ridge'
    classifiers=[SVC_un, log_un, ridge_un]
    result = pd.concat(classifiers)
    ticks=[.9, .92, .94, .96, .98, 1]
    sns.set_context("talk")
    plt.figure(figsize=(15,8))
    ax=sns.barplot('Analysis', 'Accuracy', hue='Task', data=result)
    ax.set(ylim=(.9, 1))
    ax.set_yticklabels(ticks)
    ax.set_title('Cross Validation Across Models')
    ax.legend(title='Task',loc='lower right')
    plt.savefig(outDir +'results/cv_barplot.png', bbox_inches='tight')

def ds_boxplot():
    SVC=pd.read_csv(outDir+'results/SVC/acc/DS/acc.csv')
    SVC['Analysis']='SVC'
    log=pd.read_csv(outDir+'results/logReg/acc/DS/acc.csv')
    log['Analysis']='logReg'
    ridge=pd.read_csv(outDir+'results/Ridge/acc/DS/acc.csv')
    ridge['Analysis']='Ridge'
    classifiers=[SVC, log, ridge]
    result = pd.concat(classifiers)

    plt.figure(figsize=(15,8))
    sns.set_context("talk")
    ax=sns.boxplot('Analysis', 'acc', hue='task', data=result)
    ax.set(ylim=(.3, 1.03))
    ax.axhline(.50, ls='--', color='r')
    ax.set_title('Different Subject Same Task Across Models')
    ax.legend(title='Task',loc='lower left')
    ax.set_ylabel('Accuracy')
    plt.savefig(outDir +'results/ds_boxplot.png', bbox_inches='tight')

def ss_boxplot():
    SVC=pd.read_csv(outDir+'results/SVC/acc/SS/acc.csv')
    SVC['Analysis']='SVC'
    log=pd.read_csv(outDir+'results/logReg/acc/SS/acc.csv')
    log['Analysis']='logReg'
    ridge=pd.read_csv(outDir+'results/Ridge/acc/SS/acc.csv')
    ridge['Analysis']='Ridge'
    classifiers=[SVC, log, ridge]
    result = pd.concat(classifiers)
    result['Train.Test']=result['train_task']+'.'+result['test_task']
    sns.set_context("talk")
    plt.figure(figsize=(28,8))
    colors = ["#0000CC", "#69A7EF","#FD8A07","#FCCF8C","#21B056","#84E5A8"]
    # Set color palette
    sns.set_palette(sns.color_palette(colors))
    ax=sns.boxplot(x="Analysis", y="acc", hue='Train.Test', data=result)
    ax.axhline(.50, ls='--', color='r')
    ax.set_ylabel('Accuracy', fontsize=25)
    ax.set_xlabel('Analysis', fontsize=25)
    ax.set_title('Same Subject Between Task Across Models', fontsize=30)
    ax.legend(title='Train.Test',loc='lower left')
    plt.savefig(outDir +'results/ss_boxplot.png', bbox_inches='tight')
def bs_boxplot():
    SVC=pd.read_csv(outDir+'results/SVC/acc/BS/acc.csv')
    SVC['Analysis']='SVC'
    log=pd.read_csv(outDir+'results/logReg/acc/BS/acc.csv')
    log['Analysis']='logReg'
    ridge=pd.read_csv(outDir+'results/Ridge/acc/BS/acc.csv')
    ridge['Analysis']='Ridge'
    classifiers=[SVC, log, ridge]
    result = pd.concat(classifiers)
    result['Train.Test']=result['train_task']+'.'+result['test_task']
    sns.set_context("talk")
    plt.figure(figsize=(28,14))
    colors = ["#0000CC", "#69A7EF","#FD8A07","#FCCF8C","#21B056","#84E5A8"]
    # Set color palette
    sns.set_palette(sns.color_palette(colors))
    ax=sns.boxplot(x="Analysis", y="acc", hue='Train.Test', data=result)
    ax.axhline(.50, ls='--', color='r')
    ax.set_ylabel('Accuracy', fontsize=25)
    ax.set_xlabel('Analysis', fontsize=25)
    ax.set_title('Different Subject Different Task Across Models', fontsize=30)
    ax.legend(title='Train.Test',loc='lower left')
    plt.savefig(outDir +'results/bs_boxplot.png', bbox_inches='tight')
