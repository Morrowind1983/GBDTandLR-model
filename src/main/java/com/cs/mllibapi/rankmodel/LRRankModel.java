package com.cs.mllibapi.rankmodel;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.HashMap;


//逻辑回归
public class LRRankModel implements RankModel {

    //默认的参数
    private Integer lr_NumClasses = 2;
    private Double lr_regParam = 0.01;
    private Integer lr_maxNumIterations = 1000;

    //训练好的模型
    private LogisticRegressionModel logisticRegressionModel;

    /**
     * 训练模型
     *
     * @param training 数据是已经处理好的，执行train()时，不会对label和特征进行变换
     */
    @Override
    public void train(JavaRDD<LabeledPoint> training) {
        logisticRegressionModel = null;
        LogisticRegressionWithLBFGS logisticRegressionWithLBFGS = new LogisticRegressionWithLBFGS();
        logisticRegressionWithLBFGS.setNumClasses(lr_NumClasses);
        logisticRegressionWithLBFGS.optimizer().setRegParam(lr_regParam);
        logisticRegressionWithLBFGS.optimizer().setMaxNumIterations(lr_maxNumIterations);
        logisticRegressionModel = logisticRegressionWithLBFGS.run(training.rdd());
        logisticRegressionModel.clearThreshold();   //Clears the threshold so that predict will output raw prediction scores.
    }

    /**
     * 执行预测
     *
     * @param samples 输入特征，LabeledPoints
     * @return 预测值和真实值
     */
    @Override
    public JavaRDD<Tuple2<Object, Object>> transform(JavaRDD<LabeledPoint> samples) {
        return samples.map(point -> {
            Double prediction = logisticRegressionModel.predict(point.features());
            return new Tuple2<>(prediction, point.label());
        });
    }

    /**
     * 返回参数字符串
     *
     * @return
     */
    @Override
    public String getParams() {
        String params = "";
        params += "numclasses=" + lr_NumClasses + "\n";
        params += "regparam=" + lr_regParam + "\n";
        params += "maxnumiterations=" + lr_maxNumIterations;
        return params;
    }

    /**
     * 设定参数
     *
     * @param paramSet
     */
    @Override
    public void setParams(HashMap<String, Object> paramSet) {
        lr_NumClasses = (Integer) paramSet.getOrDefault("numclasses", 2);
        lr_regParam = (Double) paramSet.getOrDefault("regparam", 0.01);
        lr_maxNumIterations = (Integer) paramSet.getOrDefault("maxnumiterations", 1000);
    }

}
