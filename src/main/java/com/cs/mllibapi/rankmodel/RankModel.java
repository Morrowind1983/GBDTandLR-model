package com.cs.mllibapi.rankmodel;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.HashMap;

//排序模型的接口
public interface RankModel extends Serializable {

    //训练
    void train(JavaRDD<LabeledPoint> training);

    //预测测试集的label
    JavaRDD transform(JavaRDD<LabeledPoint> samples);

    //返回参数
    String getParams();

    //设置参数
    void setParams(HashMap<String, Object> paramSet);
}
