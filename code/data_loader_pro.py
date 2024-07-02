
import numpy as np
import pandas as pd
import torch

torch.set_printoptions(sci_mode=False)
# camel
ff = [ 'ant','jedit','lucene', 'xerces'   ,   'velocity','xalan',         'synapse','log4j','poi',       'ivy', 'prop6','redaktor','tomcat']
folder_path = ['ant-1.7','jedit-4.1','lucene-2.4',  'xerces-1.3',          'velocity-1.6','xalan-2.6',            'synapse-1.2','log4j-1.1','poi-3.0',      'ivy-2.0','prop6','redaktor','tomcat']

ff_1= [
      [ 'jedit','lucene','xerces', 'velocity','xalan', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','lucene','xerces', 'velocity','xalan', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit', 'xerces', 'velocity','xalan', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene', 'velocity','xalan', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'xalan', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity', 'synapse','log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'log4j','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse','poi','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse', 'log4j','ivy', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse', 'log4j','poi', 'prop6','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse', 'log4j','poi', 'ivy','redaktor','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse', 'log4j','poi', 'ivy','prop6','tomcat'],
      [ 'ant','jedit','lucene','xerces', 'velocity','xalan', 'synapse', 'log4j','poi', 'ivy','prop6','redaktor']
]

folder_path1 = [
                ['jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1', 'xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','poi-3.0','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','ivy-2.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','prop6','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','redaktor','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','tomcat'],
                ['ant-1.7','jedit-4.1','lucene-2.4','xerces-1.3', 'velocity-1.6','xalan-2.6', 'synapse-1.2','log4j-1.1','poi-3.0','ivy-2.0','prop6','redaktor']
                ]

folder_path2 = [
                ['jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0', 'xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2',  'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'log4j-1.0','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','poi-2.5','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','ivy-1.4'],
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5'],                
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'], 
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'], 
                ['ant-1.6','jedit-4.0','lucene-2.2','xerces-1.2', 'velocity-1.5','xalan-2.5', 'synapse-1.1','log4j-1.0','poi-2.5','ivy-1.4'], 
                ]

def load_partition_data(ra,client_number,testname,data_dir="..."):
    test_project_idx = ff.index(testname)
    test_data_global = data_dir+testname+'/'+folder_path[test_project_idx]
    distillation_data =  '...'
    class_num = 2

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    for j,name in enumerate(folder_path1[test_project_idx]) :
        client_idx = j
        train_data_local = data_dir+ff_1[test_project_idx][j]+folder_path1[test_project_idx][j]
        train_data = torch.load(train_data_local)
        local_data_num = len(train_data)
        data_local_num_dict[client_idx] = local_data_num
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_global

    for j,name in enumerate(folder_path2[test_project_idx]) :
        client_idx = len(folder_path1[test_project_idx])+j
        train_data_local = data_dir+ff_1[test_project_idx][j]+folder_path2[test_project_idx][j]
        train_data = torch.load(train_data_local)
        local_data_num = len(train_data)
        data_local_num_dict[client_idx] = local_data_num
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_global

    return test_data_global, data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num ,distillation_data




