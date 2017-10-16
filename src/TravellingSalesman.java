
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravellingSalesman {
    /** The n value */
    //private static int N = 100;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        int n=20;
        
        StringBuilder sbSa = new StringBuilder();        
        StringBuilder sbGa = new StringBuilder();        
        StringBuilder sbMm = new StringBuilder();        
        
        sbSa.append("N"+','+"Optimal"+','+"Calls"+','+"Time"+'\n');
        sbGa.append("N"+','+"Optimal"+','+"Calls"+','+"Time"+'\n');
        sbMm.append("N"+','+"Optimal"+','+"Calls"+','+"Time"+'\n');
        
        for(int k=1; k<=6; k++) {
        	
        	int N=k*n;
        	System.out.println("N="+N);
	        double[][] points = new double[k*N][2];
	        for (int i = 0; i < points.length; i++) {
	            points[i][0] = random.nextDouble();
	            points[i][1] = random.nextDouble();   
	        }
	        // for rhc, sa, and ga we use a permutation based encoding
	        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
	        Distribution odd = new DiscretePermutationDistribution(N);
	        NeighborFunction nf = new SwapNeighbor();
	        MutationFunction mf = new SwapMutation();
	        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);	
	        
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1000);
	        
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
	        
	        sbGa.append(N);
	        sbMm.append(',');
            sbMm.append(trainingTime);	        
	        sbMm.append(',');      
	        
	
	        System.out.println("SA");
	        
//	        sbSa.append(N);
//	        sbSa.append(',');
	        
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E15, .5, hcp);
	         fit = new FixedIterationTrainer(sa, 1000);
	        
	        start = System.nanoTime();
	        fit.train();
	        end = System.nanoTime();
            
	        double optimal = ef.value(sa.getOptimal());
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
	        
	        System.out.println("GA");
	        
//	        sbSa.append(N);
//	        sbGa.append(',');
	        
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(4*N, N*3, (int) (N*0.4), gap);
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
	        sbMm.append('\n');
	        	        
	        System.out.println("Mimic");
	        
//	        sbMm.append(N);
//	        sbMm.append(',');
	        // for mimic we use a sort encoding
	        ef = new TravelingSalesmanSortEvaluationFunction(points);
	        int[] ranges = new int[N];
	        Arrays.fill(ranges, N);
	        odd = new  DiscreteUniformDistribution(ranges);
	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
	        
	        MIMIC mimic = new MIMIC((int)(4*N), (int)(2*N), pop);
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
			PrintWriter pw = new PrintWriter(new File("tspOptima4Final.csv"));
			pw.write(sbSa.toString());
	        pw.close();
	        
	        pw = new PrintWriter(new File("tspCalls4Final.csv"));
			pw.write(sbGa.toString());
	        pw.close();
	        
	        pw = new PrintWriter(new File("tspTime4Final.csv"));
			pw.write(sbMm.toString());
	        pw.close();
	        
        } catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
