package com.cs.mllibapi.rankmodel;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.configuration.FeatureType;
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.tree.model.Node;
import org.apache.spark.mllib.tree.model.Split;
import scala.Option;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

// https://www.deeplearn.me/1944.html
public class GBDTAndLRRankModel implements RankModel {

    //默认的参数
    private Integer gbdt_MaxDepth = 2;
    private Integer gbdt_numTrees = 2;
    private Integer gbdt_NumClasses = 2;
    private Integer lr_NumClasses = 2;
    private Double lr_regParam = 0.01;
    private Integer lr_maxNumIterations = 1000;

    //训练好的模型
    private GradientBoostedTreesModel gradientBoostedTreesModel;
    private LogisticRegressionModel logisticRegressionModel;

    /**
     * 训练
     *
     * @param training 数据是已经处理好的，执行train()时，不会对label和特征进行变换
     */
    @Override
    public void train(JavaRDD<LabeledPoint> training) {
        gradientBoostedTreesModel = null;
        logisticRegressionModel = null;

        //得到gbdt Model
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(gbdt_numTrees);
        //boostingStrategy.setLearningRate(gbdt_learningRate);
        //boostingStrategy.setNumIterations(gbdt_maxNumIterations);
        Strategy strategy = Strategy.defaultStrategy("Classification");
        strategy.setMaxDepth(gbdt_MaxDepth);
        strategy.setNumClasses(gbdt_NumClasses);
        boostingStrategy.setTreeStrategy(strategy);
        gradientBoostedTreesModel = GradientBoostedTrees.train(training, boostingStrategy);
        System.out.println("GBDT训练完成");

        //用gbdt model得到新特征
        JavaRDD<LabeledPoint> newFeatureDataSet = convertFeatureByGBDT(training);
        System.out.println("新特征：");
        newFeatureDataSet.take(10).forEach(s -> System.out.println(s));

        //得到LR model
        LogisticRegressionWithLBFGS logisticRegressionWithLBFGS = new LogisticRegressionWithLBFGS();
        logisticRegressionWithLBFGS.setNumClasses(lr_NumClasses);
        logisticRegressionWithLBFGS.optimizer().setRegParam(lr_regParam);
        logisticRegressionWithLBFGS.optimizer().setMaxNumIterations(lr_maxNumIterations);
        logisticRegressionModel = logisticRegressionWithLBFGS.run(newFeatureDataSet.rdd());
        logisticRegressionModel.clearThreshold();
        System.out.println("LR训练完成");
    }


    /**
     * 执行预测
     *
     * @param samples 输入特征，LabeledPoints
     * @return 预测值和真实值
     */
    @Override
    public JavaRDD<Tuple2<Object, Object>> transform(JavaRDD<LabeledPoint> samples) {
        JavaRDD<LabeledPoint> newFeatureDataSet = convertFeatureByGBDT(samples);    //转换特征
        JavaRDD<Tuple2<Object, Object>> result = newFeatureDataSet.map(point -> {   //predict
            Double prediction = logisticRegressionModel.predict(point.features());
            return new Tuple2<>(prediction, point.label());
        });
        return result;
    }

    /**
     * @return 参数字符串
     */
    @Override
    public String getParams() {
        String params = "";
        params += "gbdt_NumClasses=" + gbdt_NumClasses + "\n";
        params += "gbdt_MaxDepth=" + gbdt_MaxDepth + "\n";
        params += "gbdt_numTrees=" + gbdt_numTrees + "\n";
        params += "lr_NumClasses=" + lr_NumClasses + "\n";
        params += "lr_regParam=" + lr_regParam + "\n";
        params += "lr_maxNumIterations=" + lr_maxNumIterations;
        return params;
    }

    /**
     * @param paramSet 参数
     */
    @Override
    public void setParams(HashMap<String, Object> paramSet) {
        gbdt_NumClasses = (Integer) paramSet.getOrDefault("gbdt_numclasses", 2);
        gbdt_MaxDepth = (Integer) paramSet.getOrDefault("gbdt_maxdepth", 5);
        gbdt_numTrees = (Integer) paramSet.getOrDefault("gbdt_numtrees", 2);

        lr_NumClasses = (Integer) paramSet.getOrDefault("lr_numclasses", 2);
        lr_regParam = (Double) paramSet.getOrDefault("lr_regparam", 0.01);
        lr_maxNumIterations = (Integer) paramSet.getOrDefault("lr_maxnumiterations", 1000);
    }

    //自定义函数，得到决策树的叶子节点
    private List<Integer> getLeafNodes(Node node) {
        List<Integer> treeLeafNodes = new ArrayList<>();
        if (node.isLeaf()) {
            treeLeafNodes.add(node.id());
        } else {
            treeLeafNodes.addAll(getLeafNodes(node.leftNode().get()));
            treeLeafNodes.addAll(getLeafNodes(node.rightNode().get()));
        }
        return treeLeafNodes;
    }

    //自定义函数，预测decision tree叶子节点的值
    private Integer predictModify(Node node, DenseVector features) {
        Option<Split> split = node.split();
        if (node.isLeaf()) {
            return node.id();
        } else {
            if (split.get().featureType() == FeatureType.Continuous()) {
                if (features.apply(split.get().feature()) <= split.get().threshold()) {
                    //          println("Continuous left node")
                    return predictModify(node.leftNode().get(), features);
                } else {
                    //          println("Continuous right node")
                    return predictModify(node.rightNode().get(), features);
                }
            } else {
                if (split.get().categories().contains(features.apply(split.get().feature()))) {
                    //          println("Categorical left node")
                    return predictModify(node.leftNode().get(), features);
                } else {
                    //          println("Categorical right node")
                    return predictModify(node.rightNode().get(), features);
                }
            }
        }
    }

    //自定义函数，使用GBDT model转换特征
    private JavaRDD<LabeledPoint> convertFeatureByGBDT(JavaRDD<LabeledPoint> sample) {
        //得到决策树的叶子节点
        DecisionTreeModel[] trees = gradientBoostedTreesModel.trees();
        Integer[][] treeLeafArray = new Integer[trees.length][];    //存储叶子节点
        for (int i = 0; i < trees.length; i++) {    //找到叶子节点
            Object[] temp = getLeafNodes(trees[i].topNode()).toArray();
            treeLeafArray[i] = Arrays.copyOf(temp, temp.length, Integer[].class);
//            // test
//            System.out.println("正在打印第 " + i + " 棵树的 topnode 叶子节点");
//            for (int k = 0; k < treeLeafArray[i].length; k++) {
//                System.out.print(treeLeafArray[i][k] + " ");
//            }
//            System.out.println();
        }

        //构造新特征
        JavaRDD<LabeledPoint> newFeatures = sample.map(labeledpoint -> {
            List<Double> newFeature = new ArrayList<>();
            for (int i = 0; i < trees.length; i++) {    //每棵树
                Integer treePredict = predictModify(trees[i].topNode(), labeledpoint.features().toDense());
                //gbdt tree is binary tree
                Double[] treeArray = new Double[(trees[i].numNodes() + 1) / 2];
                for (int k = 0; k < treeLeafArray[i].length; k++) {
                    treeArray[k] = 0.0;
                    if (treeLeafArray[i][k].equals(treePredict)) {
                        treeArray[k] = 1.0;
                    }
                }
                newFeature.addAll(Arrays.asList(treeArray)); //合并特征
            }
            //转为double[]
            double[] temp = new double[newFeature.size()];
            for (int i = 0; i < newFeature.size(); i++) {
                temp[i] = newFeature.get(i).doubleValue();
            }
            return new LabeledPoint(labeledpoint.label(), new DenseVector(temp));
        });

        System.out.println("GBDT Model 转换特征完成");
        return newFeatures;
    }
}
