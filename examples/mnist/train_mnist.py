#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import data


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# 各種パラメータの定義・設定
batchsize = 100 # 確率的勾配降下法で学習させる際の1回分のbatchsize
n_epoch = 20 # 学習の繰り返し回数
n_units = 1000 # 中間層の数

# Prepare dataset
"""
Scikit LearnをつかってMNISTの手書き数字データをダウンロード
HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
"""
print('load MNIST dataset')
mnist = data.load_mnist_data()
# mnist.data : 70,000件の784次元ベクトルデータ
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255 # 0-1のデータに変換
# mnist.target : 正解データ（教師データ）
mnist['target'] = mnist['target'].astype(np.int32)

# 学習用データを N個、検証用データを残りの個数と設定
N = 60000 # 学習サンプル
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
"""
ここからモデルの定義.Chainerのクラスや関数を使う.ポイントは入出力定義
多層パーセプトロンモデルの設定
l1からl3の3層のNNのモデルの構築法
入力の手書き数字のデータがsize = 28の28*28=784次元ベクトルなので入力素子は784個
今回中間層はn_unitsで1000と指定
出力は、数字を識別するので10個
"""
model = chainer.FunctionSet(l1=F.Linear(784, n_units), # 入力 size = 28の28*28=784次元ベクトル
                            l2=F.Linear(n_units, n_units), # 中間層  n_units = 1000 次元ベクトル
                            l3=F.Linear(n_units, 10)) # 出力層 10次元ベクトル
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Neural net architecture
"""
順伝播
下記のforward()関数で定義される
NNの構造,feedforwardの部分
"""

def forward(x_data, y_data, train=True):
    """
    ポイントはVariableクラス

    各関数等の説明
    Chainerの作法で,データは配列からChainerのVariableという型（クラス）のオブジェクトに変換して使う
    """
    x, t = chainer.Variable(x_data), chainer.Variable(y_data) # Variable で使う変数を変換している
    """
    ポイントはrelu関数,dropout関数

    活性化関数はシグモイド関数ではなく、F.relu()関数が使われている.
    このF.relu()は正規化線形関数(Rectified Linear Unit function)で,f(x)=max(0,x).
    シンプルな関数のため,計算量が小さく学習スピードが速くなることが利点のよう
    このrelu()関数の出力を入力としてF.dropout()関数が使われている.
    ランダムに中間層をドロップし,過学習を防ぐことができるらしい.
    dropoutやreluは層としてではなくforwardするときに処理を加えている
    """


def forward(x_data, y_data, train=True):
    # Neural net architecture
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)

    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    # 同じ構造がもう１層あり,
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力され,出力値がyとなる
    y = model.l3(h2)
    """
    ポイントはsoftmax関数

    返り値は誤差と精度
    多クラス分類なので誤差関数としてソフトマックス関数と交差エントロピー関数を用い,誤差を導出
    F.accuracy()関数は出力と、教師データを照合して正答率(精度)を返しています。
    ソフトマックス関数を挟むことで10個の出力の総和が1となり,
    出力を確率として解釈することが可能.
    """
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# Setup optimizer
"""
モデルが決まったので訓練に移る
ここでは最適化手法としてAdamが使われている
optimizerで勾配法を選択する
"""
optimizer = optimizers.Adam()
optimizer.setup(model)

# 以上の準備から、ミニバッチ学習で手書き数字の判別を実施し、その精度を見ていく
# Learning loop
# 学習フェーズ
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    # N個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)
        # loss.backward() 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer.update()
        # optimizerはlearningのparameter情報を保持している

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    # 訓練データの誤差と,正解精度を表示
    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # code : 評価
    # evaluation
    # テストデータで誤差と、正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    # テストデータでの誤差と、正解精度を表示
    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
