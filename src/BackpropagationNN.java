
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import func.nn.backprop.*;

/**
 * An XOR test
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class BackpropagationNN {
	private static Instance[] instances = initializeInstances();
	private static DataSet trainSet;
	private static DataSet testSet;
	   
    /**
     * Tests out the perceptron with the classic xor test
     * @param args ignored
     */
    public static void main(String[] args) {
    	
    	makeTestTrainSets();
    	
        BackPropagationNetworkFactory factory = 
            new BackPropagationNetworkFactory();
//        double[][][] data = {
//            { { 0 }, { 0 } },
//            { { 0 }, { 1 } },
//            { { 0 }, { 1 } },
//        };
//        Instance[] patterns = new Instance[data.length];
//        for (int i = 0; i < patterns.length; i++) {
//            patterns[i] = new Instance(data[i][0]);
//            patterns[i].setLabel(new Instance(data[i][1]));
//        }
        BackPropagationNetwork network = factory.createClassificationNetwork(
           new int[] { 13, 14, 3, 2, 1 });
        //DataSet set = new DataSet(patterns);
        ConvergenceTrainer trainer = new ConvergenceTrainer(
               new BatchBackPropagationTrainer(trainSet, network,
                   new SumOfSquaresError(), new RPROPUpdateRule()));
        
//        for (int i = 0; i < patterns.length; i++) {
//            network.setInputValues(patterns[i].getData());
//            network.run();
//            System.out.println("~~");
//            System.out.println(patterns[i].getLabel());
//            System.out.println(network.getOutputValues());
//        }
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        trainer.train();
        System.out.println("Convergence in " 
            + trainer.getIterations() + " iterations");
        
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);            
        
//        System.out.println(nnop[i].getTotalCalls());          
        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < trainSet.getInstances().length; j++) {
//        for(int j = 0; j < 5; j++) {
            network.setInputValues(trainSet.getInstances()[j].getData());
            network.run();

            actual = Double.parseDouble(trainSet.getInstances()[j].getLabel().toString());
            predicted = Double.parseDouble(network.getOutputValues().toString());
//            System.out.println(actual);
//            System.out.println(predicted);
            
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        
        double error = incorrect/(correct+incorrect)*100;
        System.out.println(error);
        
        correct = 0;
        incorrect = 0;
        start = System.nanoTime();
        for(int j = 0; j < testSet.getInstances().length; j++) {
//        for(int j = 0; j < 5; j++) {
            network.setInputValues(testSet.getInstances()[j].getData());
            network.run();

            actual = Double.parseDouble(testSet.getInstances()[j].getLabel().toString());
            predicted = Double.parseDouble(network.getOutputValues().toString());
//            System.out.println(actual);
//            System.out.println(predicted);
            
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        
        error = incorrect/(correct+incorrect)*100;
        System.out.println(error);

        
//        resultsTrain +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
//                    "\nIncorrectly classified " + incorrect + " instances.\nPercent incorrectly classified: "
//                    + df.format(incorrect/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
//                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds TotalFunctionCalls=+"+nnop[i].getTotalCalls()+"\n";
//        
//        nnop[i].clearCount();
//        
//        correct = 0;
//        incorrect = 0;
//        start = System.nanoTime();
//        for(int j = 0; j < testSet.getInstances().length; j++) {
//            networks[i].setInputValues(testSet.getInstances()[j].getData());
//            networks[i].run();
//
//            predicted = Double.parseDouble(testSet.getInstances()[j].getLabel().toString());
//            actual = Double.parseDouble(networks[i].getOutputValues().toString());
//
//            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
//
//        }
//        end = System.nanoTime();
//        testingTime = end - start;
//        testingTime /= Math.pow(10,9);
//
//        resultsTest +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
//                    "\nIncorrectly classified " + incorrect + " instances.\nPercent incorrectly classified: "
//                    + df.format(incorrect/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
//                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
//        
//        System.out.println("Train Results");
//        System.out.println(resultsTrain);
//        System.out.println("Test Results");
//        System.out.println(resultsTest);
    
    }
    
    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[3012][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("AnnNormalized.txt")));
            //System.out.println(attributes.length);
            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[13]; // 13 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 13; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
              //      System.out.println(i+" "+j+" "+attributes[i][0][j]);
                }
                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];
        //System.out.println(instances.length);
        for(int i = 0; i < instances.length; i++) {
          //  System.out.println(i);
            //if(attributes[i][0]==null)
            //System.out.println("NULL");
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }
        
        return instances;
    }

    public static void makeTestTrainSets() {

        List<Instance> instancesList = new ArrayList<>();

        for (Instance instance: instances) {
          instancesList.add(instance);
        }

        Random rand = new Random(0xABCDEF);
        Collections.shuffle(instancesList, rand);

        int cutoff = (int) (instancesList.size() * 0.7);

        List<Instance> trainInstances = instancesList.subList(0, cutoff);
        List<Instance> testInstances = instancesList.subList(cutoff, instancesList.size());

        Instance[] arr_trn = new Instance[trainInstances.size()];
        trainSet = new DataSet(trainInstances.toArray(arr_trn));

//        System.out.println("Train Set ")

        Instance[] arr_tst = new Instance[testInstances.size()];
        testSet = new DataSet(testInstances.toArray(arr_tst));

      }

}