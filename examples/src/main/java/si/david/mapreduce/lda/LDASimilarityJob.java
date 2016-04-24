package si.david.mapreduce.lda;

import org.apache.commons.clilda.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

/**
 * Created by dstarina on 2/23/16.
 */
public class LDASimilarityJob extends AbstractJob {

        public static final String ANSI_RESET = "\u001B[0m";
        public static final String ANSI_RED = "\u001B[31m";

        static String ldaOutputPath ="";
        static int similaritiesForDocument = 0; // -1 = all documents;
        static int numSimilarDocuments = -1; // number of similar documents to find;

        private static final Logger log = LoggerFactory.getLogger(Job.class);
        //static int numTopics = 20;


        public static void main(String args[]) throws Exception {
                Option fileLocationOpt = Option.builder("f")
                        .hasArg()
                        .argName("location")
                        .longOpt("file")
                        .desc("file location")
                        .required()
                        .build();
                Option outFileLocationOpt = Option.builder("o")
                        .hasArg()
                        .argName("folder")
                        .longOpt("output")
                        .desc("output location (reserved for future use)")
                        .build();
                Option docNumberOpt = Option.builder("d")
                        .hasArg()
                        .argName("number")
                        .longOpt("document")
                        .desc("number of document to compare")
                        .build();
                Option numSimilaritiesOpt = Option.builder("n")
                        .hasArg()
                        .argName("number")
                        .longOpt("numSimilarities")
                        .desc("number of similar documents to find")
                        .build();

                Options options = new Options();
                options.addOption(fileLocationOpt);
                options.addOption(outFileLocationOpt);
                options.addOption(docNumberOpt);
                options.addOption(numSimilaritiesOpt);


                CommandLineParser parser = new DefaultParser();
                CommandLine line = parser.parse( options, args );

                String baseFileLocation="";
                if(line.hasOption("f")) {
                        baseFileLocation = line.getOptionValue("f");
                } else {
                        HelpFormatter formatter = new HelpFormatter();
                        String fileName="mahout-lda-job.jar";
                        formatter.printHelp( "Location is required. Usage: hadoop jar "+fileName+" si.david.mapreduce.lda.LDASimilarityJob", options );

                        return;
                }

                if(line.hasOption(outFileLocationOpt.getOpt())) {
                        ldaOutputPath =line.getOptionValue(outFileLocationOpt.getOpt());
                        if(!ldaOutputPath.endsWith("/")) {
                                ldaOutputPath = ldaOutputPath +"/";
                        }
                } else {
                        ldaOutputPath=baseFileLocation;
                        if(!ldaOutputPath.endsWith("/")) {
                                ldaOutputPath = ldaOutputPath +"/";
                        }
                        ldaOutputPath+="output";
                }
                if(line.hasOption(docNumberOpt.getOpt())) {
                        similaritiesForDocument = Integer.parseInt(line.getOptionValue(docNumberOpt.getOpt()));
                }
                if(line.hasOption(numSimilaritiesOpt.getOpt())) {
                        numSimilarDocuments = Integer.parseInt(line.getOptionValue(numSimilaritiesOpt.getOpt()));
                }

                Path output = new Path(ldaOutputPath);
                Configuration conf = new Configuration();
                HadoopUtil.delete(conf, output);
                String[] similarityArgs = {"-DbaseFileLocation=" + baseFileLocation};

                ToolRunner.run(new LDASimilarityJob(), similarityArgs);
                System.out.println("done");
        }

        public int run(String[] arg0) throws Exception {
                Configuration conf = getConf();
                String baseFileLocation = conf.get("baseFileLocation");

                ArrayList<Pair<String, Vector>> a = new ArrayList<Pair<String, Vector>>();
                //SequenceFile.Reader.file()

                Path files = new Path(baseFileLocation);
                FileSystem fs = files.getFileSystem(conf);
                Path[] outfiles = DFUtils.listOutputFiles(fs, files);
                for (Path path : outfiles) {

                        //SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(new Path(baseFileLocation), true, conf);
                        SequenceFileIterable<Writable, Writable> iterable = new SequenceFileIterable<Writable, Writable>(path, true, conf);
                        Iterator<Pair<Writable, Writable>> iterator = iterable.iterator();
                        //long i = 0;
                        while (iterator.hasNext()) {
                                Pair<Writable, Writable> record = iterator.next();
                                Writable keyWritable = record.getFirst();
                                Writable valueWritable = record.getSecond();

                                Vector vector;
                                try {
                                        vector = ((VectorWritable) valueWritable).get();
                                } catch (ClassCastException e) {
                                        if (valueWritable instanceof WeightedPropertyVectorWritable) {
                                                vector = ((WeightedPropertyVectorWritable) valueWritable).getVector();
                                        } else {
                                                throw e;
                                        }
                                }
                                a.add(new Pair<String, Vector>(keyWritable.toString(), vector));
                        }
                }

                ArrayList<ValueComparablePair<String, Double>> distances = new ArrayList<ValueComparablePair<String, Double>>();
                Vector firstVector = a.get(similaritiesForDocument).getSecond();
                for(Pair<String, Vector> p: a) {
                        distances.add(new ValueComparablePair<String, Double>(p.getFirst(), cosineSimilarity(firstVector, p.getSecond())));
                }

                Collections.sort(distances);

                System.out.println(ANSI_RED+"Output:"+ANSI_RESET);
                for(int i = 0; i < numSimilarDocuments; i++) {
                        System.out.print("("+distances.get(distances.size()-1-i).getFirst().toString()+","+distances.get(distances.size()-1-i).getSecond().toString()+") ");
                }

                return 0;
        }


        /*
        private double euclidianDistance(double[] d, double[] d0) {
                double med = 0;
                double sum = 0;
                for (int i = 1; i < d.length; i++)
                {
                        sum += (d[i]-d0[i])* (d[i]-d0[i]);
                }
                med = Math.sqrt(sum);
                return med;
        }*/


        // http://computergodzilla.blogspot.si/2013/07/how-to-calculate-tf-idf-of-document.html
        public double cosineSimilarity(Vector vector1, Vector vector2) {
                double dotProduct = 0;
                double length1 = 0;
                double length2 = 0;
                double cosineSimilarity = 0;

                /*
                for (int i = 0; i < vector1.size(); i++) //vector1 and vector2 must be of same length
                {
                        dotProduct += vector1[i] * vector2[i];  //a.b
                        length1 += vector1[i]*vector1[i];  //(a^2)
                        length2 += vector2[i]*vector2[i]; //(b^2)
                }
                */

                dotProduct=vector1.dot(vector2);
                length1=vector1.getLengthSquared();
                length2=vector2.getLengthSquared();

                length1 = Math.sqrt(length1);//sqrt(a^2)
                length2 = Math.sqrt(length2);//sqrt(b^2)

                if (length1 != 0.0 | length2 != 0.0) {
                        cosineSimilarity = dotProduct / (length1 * length2);
                } else {
                        return 0.0;
                }
                return cosineSimilarity;
        }


}
