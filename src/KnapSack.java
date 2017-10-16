import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapSack {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 60;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	
    	StringBuilder sbSa = new StringBuilder();        
        StringBuilder sbGa = new StringBuilder();        
        StringBuilder sbMm = new StringBuilder();        
        
        sbSa.append("N"+','+"Optimal"+','+"Optimal"+','+"Optimal"+'\n');
        sbGa.append("N"+','+"Calls"+','+"Calls"+','+"Calls"+'\n');
        sbMm.append("N"+','+"Time"+','+"Time"+','+"Time"+'\n');
        
        for(int k=1; k<=6; k++) {
        	
        	int N = k*NUM_ITEMS;
        	System.out.println(N);
        	
	        int[] copies = new int[N];
	        Arrays.fill(copies, COPIES_EACH);
	        double[] values = new double[N];
	        double[] weights = new double[N];
	        for (int i = 0; i < N; i++) {
	            values[i] = random.nextDouble() * MAX_VALUE;
	            weights[i] = random.nextDouble() * MAX_WEIGHT;
	        }
	        int[] ranges = new int[N];
	        Arrays.fill(ranges, COPIES_EACH + 1);
	
	        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, k*MAX_KNAPSACK_WEIGHT, copies);
	        Distribution odd = new DiscreteUniformDistribution(ranges);
	        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
	
	        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
	        CrossoverFunction cf = new UniformCrossOver();
	        Distribution df = new DiscreteDependencyTree(.1, ranges);
	
	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);	        
	        
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 500);
	        double start = System.nanoTime();
	        fit.train();
	        double end = System.nanoTime();
	        System.out.println(ef.value(rhc.getOptimal()));
	        
	        sbSa.append(N);
	        sbSa.append(',');
	        sbSa.append(ef.value(rhc.getOptimal()));
	        sbSa.append(',');
	        
	        int calls = ef.getTotalCalls();
	        System.out.println(calls);
	        ef.clearCount();
	        sbGa.append(N);
	        sbGa.append(',');
	        sbGa.append(calls);
	        sbGa.append(',');
	        
	        double trainingTime = end - start;
	        trainingTime /= Math.pow(10,9);
	        System.out.println(trainingTime);
	        
	        sbMm.append(N);
	        sbMm.append(',');
            sbMm.append(trainingTime);	        
	        sbMm.append(',');      
	        
	        System.out.println("SA");
	        
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E15, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 1000);
	        
	        start = System.nanoTime();
	        fit.train();
	        end = System.nanoTime();
	        
	        double optimal = ef.value(sa.getOptimal());
	        System.out.println(optimal);
//	        sbSa.append(N);
	        sbSa.append(',');
	        sbSa.append(optimal);
	        sbSa.append(',');
	        
	        calls = ef.getTotalCalls();
	        System.out.println(calls);
	        ef.clearCount();
//	        sbGa.append(N);
	        sbGa.append(',');
	        sbGa.append(calls);
	        sbGa.append(',');
	        
	        trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            System.out.println(trainingTime);
//            sbMm.append(N);
	        sbMm.append(',');
            sbMm.append(trainingTime);	        
	        sbMm.append(',');      

	        System.out.println("GA");	        	        

	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int)(2*N), (int)(N), (int)(0.3*N), gap);
	        fit = new FixedIterationTrainer(ga, 1000);
	        start = System.nanoTime();
	        fit.train();
	        end = System.nanoTime();	        
	        
	        optimal = ef.value(ga.getOptimal());
	        System.out.println(optimal);	
	        
	        sbSa.append(optimal);
	        sbSa.append(',');
	        
	        calls = ef.getTotalCalls();
	        System.out.println(calls);
	        ef.clearCount();
	        
	        sbGa.append(calls);
	        sbGa.append(',');
	        
	        trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
	        System.out.println(trainingTime);
	        
	        sbMm.append(trainingTime);
	        sbMm.append(',');
	        
	        System.out.println("Mimic");     	        
	        
	        MIMIC mimic = new MIMIC((int)(2*N), (int)(N), pop);
	        fit = new FixedIterationTrainer(mimic, 1000);
	        start = System.nanoTime();
	        fit.train();
	        end = System.nanoTime();
	        	       
	        optimal = ef.value(mimic.getOptimal());
	        System.out.println(optimal);
	        
	        sbSa.append(optimal);
	        sbSa.append('\n');
	        
	        calls = ef.getTotalCalls();
	        System.out.println(calls);
	        ef.clearCount();
	        
	        sbGa.append(calls);
	        sbGa.append('\n');
	        
	        trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
	        System.out.println(trainingTime);
	        
	        sbMm.append(trainingTime);
	        sbMm.append('\n');
        }
        try {
			PrintWriter pw = new PrintWriter(new File("KnapSackOptimalFinal4-1.csv"));			
			pw.write(sbSa.toString());  
			pw.close();
			
			pw = new PrintWriter(new File("KnapSackCallsFinal4-1.csv"));
	        pw.write(sbGa.toString());
	        pw.close();
	        
	        pw = new PrintWriter(new File("KnapSackTimeFinal4-1.csv"));
	        pw.write(sbMm.toString());	        
	        pw.close();
	        System.out.println("Done");
	        
        } catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
 
    }
}


    