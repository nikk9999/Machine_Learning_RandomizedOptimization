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
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnes {
    /** The n value */
    private static final int N = 20;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
//        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200);
//        fit.train();
//        System.out.println(ef.value(rhc.getOptimal()));
        try {
			PrintWriter pw = new PrintWriter(new File("Result.csv"));
		
	        StringBuilder sb = new StringBuilder();
	        sb.append("SA");
	        sb.append('\n');
	        
	        
	        
	        FixedIterationTrainer fit;
	        System.out.println("SA");
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
	        for(int i=0; i<10; i++) {
	        	fit = new FixedIterationTrainer(sa, 20*(i+1));
	            fit.train();
	            System.out.println(ef.value(sa.getOptimal()));
	            System.out.println(ef.getTotalCalls());
	            
	            sb.append(ef.value(sa.getOptimal()));
	            sb.append(',');
	            sb.append(ef.getTotalCalls());
	            sb.append('\n');
	            ef.clearCount();
	        }
	        pw.write(sb.toString());
	        pw.close();
	        System.out.println("-----------");
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
	        for(int i=0; i<10; i++) {
		        fit = new FixedIterationTrainer(ga, 20*(i+1));
		        fit.train();
		        System.out.println(ef.value(ga.getOptimal()));
		        System.out.println(ef.getTotalCalls());
	            ef.clearCount();
		    }
	        System.out.println("-----------");
	        MIMIC mimic = new MIMIC(50, 10, pop);
	        for(int i=0; i<10; i++) {
		        fit = new FixedIterationTrainer(mimic, 20*(i+1));
		        fit.train();
		        System.out.println(ef.value(mimic.getOptimal()));
		        System.out.println(ef.getTotalCalls());
	            ef.clearCount();        
	        }
        } catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}