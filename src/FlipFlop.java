import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlop {
    /** The n value */
    private static final int n = 50;
    
    public static void main(String[] args) {
    	
    	StringBuilder sbSa = new StringBuilder();        
        StringBuilder sbGa = new StringBuilder();        
        StringBuilder sbMm = new StringBuilder();        
        
        sbSa.append("N"+','+"Optimal"+','+"Optimal"+','+"Optimal"+'\n');
        sbGa.append("N"+','+"Calls"+','+"Calls"+','+"Calls"+'\n');
        sbMm.append("N"+','+"Time"+','+"Time"+','+"Time"+'\n');
        
        for(int k=0; k<6; k++) {
	        	        	      
	    	int N=n+k*50;
	    	System.out.println(N);
        	int[] ranges = new int[N];
	        Arrays.fill(ranges, 2);
	        EvaluationFunction ef = new FlipFlopEvaluationFunction();
	        Distribution odd = new DiscreteUniformDistribution(ranges);
	        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
	        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
	        CrossoverFunction cf = new SingleCrossOver();
	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
	        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
	        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
	        
	        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
	        
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
	        
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, 0.95, hcp);
	        fit = new FixedIterationTrainer(sa, 200000);
	        
	        start = System.nanoTime();
	        fit.train();
	        end = System.nanoTime();
	        
	        double optimal = ef.value(sa.getOptimal());
	        System.out.println(optimal);
//	        sbSa.append(N);
//	        sbSa.append(',');
	        sbSa.append(optimal);
	        sbSa.append(',');
	        
	        calls = ef.getTotalCalls();
	        System.out.println(calls);
	        ef.clearCount();
//	        sbGa.append(N);
//	        sbGa.append(',');
	        sbGa.append(calls);
	        sbGa.append(',');
	        
	        trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            System.out.println(trainingTime);
//            sbMm.append(N);
//	        sbMm.append(',');
            sbMm.append(trainingTime);	        
	        sbMm.append(',');      
	        
	        System.out.println("GA");	        	        
	        
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm((int)(4*N), (int)(3*N), (int) (0.4*N), gap);
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
	        
	        MIMIC mimic = new MIMIC((int) (4*N), (int)(2*N), pop);
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
			PrintWriter pw = new PrintWriter(new File("flipFlopOptimal4Final-1.csv"));			
			pw.write(sbSa.toString());  
			pw.close();
			
			pw = new PrintWriter(new File("flipFlopCalls4Final-1.csv"));
	        pw.write(sbGa.toString());
	        pw.close();
	        
	        pw = new PrintWriter(new File("flipFlopTime4Final-1.csv"));
	        pw.write(sbMm.toString());	        
	        pw.close();
	        System.out.println("Done");
	        
        } catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    
     }
}