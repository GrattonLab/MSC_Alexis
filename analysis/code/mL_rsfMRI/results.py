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
outDir = thisDir + 'output/'
#Subjects and tasks
taskList=['mixed', 'motor','mem']
subList=['MSC01','MSC02','MSC03','MSC04','MSC05','MSC06','MSC07','MSC10']
def plotACC(df, classifier, analysis):
    if analysis == 'CV':
        print('task by subject')
        ax=sns.heatmap(df, annot=True, vmin=.5, vmax=1)
        fig=ax.get_figure()
        fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/acc_heatmap.png', bbox_inches='tight')
    elif analysis=='SS':
        print('task by subject')
        grouped_df=df.groupby('test_task')
        for task in taskList:
            task_df=grouped_df.get_group(task)
            task_df.drop(columns=['test_task'], inplace=True)
            task_df=task_df.pivot(index='sub', columns='train_task', values='acc')
            plt.figure()
            ax=sns.heatmap(task_df, annot=True, vmin=.5, vmax=1)
            ax.set_title('Testing ' + task)
            ax.set_xlabel('Training Variables')
            fig=ax.get_figure()
            fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/'+task+'_acc_heatmap.png', bbox_inches='tight')
    elif analysis =='DS':
        print('subject by subject')
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
            fig=ax.get_figure()
            fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/'+task+'_acc_heatmap.png', bbox_inches='tight')
    elif analysis =='BS':
        print('subject by subject per train task split')
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
                    fig=ax.get_figure()
                    fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/'+train_task+'_'+test_task+'_acc_heatmap.png', bbox_inches='tight')

    else:
        print('not enough information')

def statsACC(df, classifier, analysis):
    if analysis=='CV':
        print('cross validation')
        mu=df.mean()
        sd=df.std()
        stats=pd.DataFrame({'Mean':mu, 'Std':sd})
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='SS':
        print('same sub')
        df.drop(['sub'], axis=1, inplace=True)
        stats=df.groupby(['test_task', 'train_task']).mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby(['test_task', 'train_task']).std()
        stats['Std']=sd['acc']
        stats.reset_index(inplace=True)
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='DS':
        print('diff sub')
        df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        stats=df.groupby('task').mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby('task').std()
        stats['Std']=sd['acc']
        stats.to_csv(outDir+ 'results/' +classifier+'/acc/'+analysis+'/stats.csv', index=True)
    elif analysis=='BS':
        print('diff sub diff task')
        df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        stats=df.groupby(['test_task', 'train_task']).mean()
        stats.rename(columns={'acc':'Mean'}, inplace=True)
        sd=df.groupby(['test_task', 'train_task']).std()
        stats['Std']=sd['acc']
        stats.reset_index(inplace=True)
    else:
        print('not enough information')
def boxACC(df, classifier, analysis):
    if analysis=='CV':
        print('cross validation')
    elif analysis=='SS':
        print('same sub')
        df.drop(['sub'], axis=1, inplace=True)
        plt.figure(figsize=(15,8))
        ax=sns.boxplot(x='test_task', y='acc', hue='train_task', data=df)
        ax.axhline(.50, ls='--', color='r')
        ax.set_title('Same Sub Between Task')
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Train Task',loc='upper right')
        fig=ax.get_figure()
        fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')

    elif analysis=='DS':
        print('diff sub')
    elif analysis=='BS':
        print('diff sub diff task')
        df.drop(['train_sub', 'test_sub'], axis=1, inplace=True)
        plt.figure(figsize=(15,8))
        ax=sns.boxplot(x='test_task', y='acc', hue='train_task', data=df)
        ax.axhline(.50, ls='--', color='r')
        ax.set(ylim=(.4))
        ax.set_title('Between Sub Between Task')
        ax.set_xlabel('Test Task')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Train Task',loc='upper right')
        fig.savefig(outDir +'images/'+classifier+'/acc/'+analysis+'/boxplot.png', bbox_inches='tight')
    else:
        print('not enough information')
