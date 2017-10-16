import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;


/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class NeuralNetwork {
    private static Instance[] instances = initializeInstances();
    
    private static int inputLayer = 13,  outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    //private static DataSet set = new DataSet(instances);
    private static DataSet trainSet;
    private static DataSet testSet;
    
    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String resultsTrain = "";
    private static String resultsTest = "";

    private static DecimalFormat df = new DecimalFormat("0.000");
    //private static ConvergenceTrainer[] ct = new ConvergenceTrainer[3];

    public static void main(String[] args) {
    	
    	makeTestTrainSets();
        
    	for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, 14,3,2, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainSet, networks[i], measure);
            
        }
        
        int N = trainSet.size();
        
        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(100, 0.95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm((int) (N), (int) (0.2*N), (int) (0.15*N), nnop[2]);              
                
        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);            
            
//            System.out.println(nnop[i].getTotalCalls());
            
            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < trainSet.getInstances().length; j++) {
                networks[i].setInputValues(trainSet.getInstances()[j].getData());
                networks[i].run();

                actual = Double.parseDouble(trainSet.getInstances()[j].getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
            
            
            resultsTrain +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent incorrectly classified: "
                        + df.format(incorrect/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds TotalFunctionCalls=+"+nnop[i].getTotalCalls()+"\n";
            
            nnop[i].clearCount();
            
            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < testSet.getInstances().length; j++) {
                networks[i].setInputValues(testSet.getInstances()[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testSet.getInstances()[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            resultsTest +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent incorrectly classified: "
                        + df.format(incorrect/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }
        System.out.println("Train Results");
        System.out.println(resultsTrain);
        System.out.println("Test Results");
        System.out.println(resultsTest);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
//        System.out.println("\nError results for " + oaName + "\n---------------------------");
        
        StringBuilder sbTrainError = new StringBuilder();    
        sbTrainError.append("Iteration"+','+"Error"+'\n');
        StringBuilder sbTestError = new StringBuilder();
        sbTestError.append("Iteration"+','+"Error"+'\n');
        
        double start = System.nanoTime(), end, time;
        int iteration=1;
        while(iteration>0) {
        	
            oa.train();

            double predicted, actual, correct=0, incorrect=0;
            
            for(int j = 0; j < trainSet.getInstances().length; j++) {
            	network.setInputValues(instances[j].getData());
            	network.run();
            	
                actual = Double.parseDouble(trainSet.getInstances()[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());
             
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                
            }
            double errorRate = incorrect/(correct+incorrect)*100;
//            System.out.println("ErrorRate="+df.format(errorRate));
            sbTrainError.append(iteration);
            sbTrainError.append(',');
            sbTrainError.append(df.format(errorRate));
            sbTrainError.append('\n');
            
            incorrect=0; correct=0;
            for(int j = 0; j < testSet.getInstances().length; j++) {
            	network.setInputValues(instances[j].getData());
            	network.run();
            	
                actual = Double.parseDouble(testSet.getInstances()[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());
             
                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            errorRate = incorrect/(correct+incorrect)*100;
//            System.out.println("ErrorRate="+df.format(errorRate));
            sbTestError.append(iteration);
            sbTestError.append(',');
            sbTestError.append(df.format(errorRate));
            sbTestError.append('\n');
            
            end = System.nanoTime();
            time = end - start;
            time /= Math.pow(10,9);
            
            if(time>1200)
            	break;
            else
            	iteration++;
            
        }
        try {
			PrintWriter pw = new PrintWriter(new File(oaName+"TrainErrorNN2.csv"));			
			pw.write(sbTrainError.toString());  
			pw.close();
			
			pw = new PrintWriter(new File(oaName+"TestErrorNN2.csv"));			
			pw.write(sbTestError.toString());  
			pw.close();
        } 
        catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
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
