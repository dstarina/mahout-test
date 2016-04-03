package si.david.mapreduce.lda;

import org.apache.commons.clilda.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.SequenceFilesFromDirectory;
import org.apache.mahout.utils.vectors.RowIdJob;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by dstarina on 2/8/16.
 */

public class LDAJob extends AbstractJob {
        public static final String ANSI_RESET = "\u001B[0m";
        public static final String ANSI_RED = "\u001B[31m";
        public static final String ANSI_GREEN = "\u001B[32m";
        public static final String ANSI_YELLOW = "\u001B[33m";
        public static final String ANSI_BLUE = "\u001B[34m";


        private static final Logger log = LoggerFactory.getLogger(Job.class);
        static int numTopics = 100;

        // naj bi bilo okoli 50/numTopics
        //static double docTopicSmoothening = 0.0001;
        //static double termTopicSmoothening = 0.0001;
        static double docTopicSmoothening = 50/numTopics; // alpha
        static double termTopicSmoothening = 50/numTopics; // eta
        // da pohitrim testiranje
        static int maxIter = 3;
        //static int maxIter = 10;
        static int iterationBlockSize = 1;
        static double convergenceDelta = 0;
        //static float testFraction = 0.0f;
        static float testFraction = 0.1f;
        static int numTrainThreads = 4;
        static int numUpdateThreads = 1;
        //static int maxItersPerDoc = 10;
        // da pohitrim testiranje
        static int maxItersPerDoc = 3;
        static int numReduceTasks = 10;
        static boolean backfillPerplexity = false;

        static int startAtStep=1;
        static boolean singleStep = false;
        static boolean printAll=false;

        static String ldaSplitSize="10485760";
        static String ldaOutputPath ="";
        static boolean useTFIDF = false;
        static int maxDFPercent = 85;
        static int minDF = 2;

        public static void main(String args[]) throws Exception {

                Option fileLocationOpt = Option.builder("f")
                        .hasArg()
                        .argName("location")
                        .longOpt("file")
                        .desc("file location")
                        .required()
                        .build();
                Option startAtStepOpt = Option.builder("s")
                        .hasArg()
                        .argName("step")
                        .longOpt("startAtStep")
                        .desc("start at step:\n"+
                                "\t1-beginning (convert the directory into seqFile)\n" +
                                "\t2-generate TF (converting the seq to vector)\n" +
                                "\t3-convert TF for LDA input\n" +
                                "\t4-LDA (Run the LDA algo)\n" +
                                "\t5-output (vectordump topic-term)")
                        .build();
                Option executeSingleStepOpt = new Option("st", "singleStep", false, "execute single step");
                Option numTopicsOpt = Option.builder("t")
                        .hasArg()
                        .argName("number")
                        .longOpt("topics")
                        .desc("number of topics (default:"+LDAJob.numTopics+")")
                        .build();
                Option numIterationsOpt = Option.builder("i")
                        .hasArg()
                        .argName("number")
                        .longOpt("iterations")
                        .desc("number of iterations (default: "+LDAJob.maxIter+")")
                        .build();
                Option numIterationsPerDocOpt = Option.builder("d")
                        .hasArg()
                        .argName("number")
                        .longOpt("iterationsPerDoc")
                        .desc("number of iterations per document (default: "+LDAJob.maxItersPerDoc+")")
                        .build();
                Option blockSizeOpt = Option.builder("b")
                        .hasArg()
                        .argName("number")
                        .longOpt("blockSize")
                        .desc("block size - how often to check perplexity (default: "+LDAJob.iterationBlockSize +")")
                        .build();
                Option trainThreadsOpt = Option.builder("tr")
                        .hasArg()
                        .argName("number")
                        .longOpt("trainThreads")
                        .desc("number of train threads (default: "+LDAJob.numTrainThreads+")")
                        .build();
                Option updateThreadsOpt = Option.builder("u")
                        .hasArg()
                        .argName("number")
                        .longOpt("updateThreads")
                        .desc("number of update threads (default: "+LDAJob.numUpdateThreads+")")
                        .build();
                Option testFractionOpt = Option.builder("tf")
                        .hasArg()
                        .argName("proportion")
                        .longOpt("testFraction")
                        .desc("test fraction (default: "+LDAJob.testFraction+")")
                        .build();
                Option convergenceDeltaOpt = Option.builder("cd")
                        .hasArg()
                        .argName("delta")
                        .longOpt("convergenceDelta")
                        .desc("convergence delta (default: "+LDAJob.convergenceDelta+")")
                        .build();
                Option ldaSplitSizeOpt = Option.builder("ls")
                        .hasArg()
                        .argName("size in bytes")
                        .longOpt("ldaSplitSize")
                        .desc("LDA split size in bytes (default: "+LDAJob.ldaSplitSize+")")
                        .build();
                Option numReduceTasksOpt = Option.builder("nr")
                        .hasArg()
                        .argName("number")
                        .longOpt("numReduceTasks")
                        .desc("Number of reduce tasks (default: "+LDAJob.numReduceTasks+")")
                        .build();
                Option backfillPerplexityOpt = new Option("bf", "backfillPerplexity", false, "backfill perplexity");
                Option printAllOpt = new Option("pa","print all results");
                Option outFileLocationOpt = Option.builder("o")
                        .hasArg()
                        .argName("folder")
                        .longOpt("output")
                        .desc("output folders location")
                        .build();
                Option useTFIDFOpt = new Option("ti", "tfidf", false, "use TF-IDF weight instead of TF");
                Option maxDFPercentOpt = Option.builder("mxd")
                        .hasArg()
                        .argName("percent")
                        .longOpt("maxDFPercent")
                        .desc("max % od docs for document frequency - removes terms, used in more than maxDFPercent% documents (default: "+maxDFPercent+")")
                        .build();
                Option minDFOpt = Option.builder("mnd")
                        .hasArg()
                        .argName("number of documents")
                        .longOpt("minDF")
                        .desc("min number od docs for document frequency - removes terms, used in less than minDF documents (default: "+minDF+")")
                        .build();
                Option alphaOpt = Option.builder("a")
                        .hasArg()
                        .argName("decimal")
                        .longOpt("alpha")
                        .desc("alpha parameter - documents per topic smoothing (default: (number of topics)/50)")
                        .build();
                Option etaOpt = Option.builder("e")
                        .hasArg()
                        .argName("decimal")
                        .longOpt("eta")
                        .desc("eta parameter - terms per topic smoothing (default: (number of topics)/50)")
                        .build();


                Options options = new Options();
                options.addOption(fileLocationOpt);
                options.addOption(startAtStepOpt);
                options.addOption(executeSingleStepOpt);
                options.addOption(numTopicsOpt);
                options.addOption(numIterationsOpt);
                options.addOption(numIterationsPerDocOpt);
                options.addOption(blockSizeOpt);
                options.addOption(trainThreadsOpt);
                options.addOption(updateThreadsOpt);
                options.addOption(testFractionOpt);
                options.addOption(convergenceDeltaOpt);
                options.addOption(backfillPerplexityOpt);
                options.addOption(printAllOpt);
                options.addOption(ldaSplitSizeOpt);
                options.addOption(numReduceTasksOpt);
                options.addOption(outFileLocationOpt);
                options.addOption(useTFIDFOpt);
                options.addOption(maxDFPercentOpt);
                options.addOption(minDFOpt);
                options.addOption(alphaOpt);
                options.addOption(etaOpt);

                if(args.length == 0) {
                        HelpFormatter formatter = new HelpFormatter();
                        formatter.printHelp( "hadoop jar mahout-lda-job.jar si.david.mapreduce.lda.LDAJob", options );
                } else {
                        CommandLineParser parser = new DefaultParser();
                        CommandLine line = parser.parse( options, args );

                        String baseFileLocation="";
                        if(line.hasOption("f")) {
                                baseFileLocation = line.getOptionValue("f");
                        } else {
                                HelpFormatter formatter = new HelpFormatter();
                                String fileName="mahout-lda-job.jar";
                                formatter.printHelp( "Location is required. Usage: hadoop jar "+fileName+" si.david.mapreduce.lda.LDAJob", options );

                                return;
                        }

                        if(line.hasOption("s")) {
                                startAtStep = Integer.parseInt(line.getOptionValue("s"));
                        }

                        if(line.hasOption(outFileLocationOpt.getOpt())) {
                                ldaOutputPath =line.getOptionValue(outFileLocationOpt.getOpt());
                                if(!ldaOutputPath.endsWith("/")) {
                                        ldaOutputPath = ldaOutputPath +"/";
                                }
                        }

                        if(line.hasOption(numTopicsOpt.getOpt())) {
                                numTopics = Integer.parseInt(line.getOptionValue(numTopicsOpt.getOpt()));
                                docTopicSmoothening = 50/numTopics;
                                termTopicSmoothening = 50/numTopics;
                        }
                        if (line.hasOption(alphaOpt.getOpt())) {
                                docTopicSmoothening=Double.parseDouble(line.getOptionValue(alphaOpt.getOpt()));
                        }
                        if (line.hasOption(etaOpt.getOpt())) {
                                termTopicSmoothening=Double.parseDouble(line.getOptionValue(etaOpt.getOpt()));
                        }

                        singleStep=line.hasOption(executeSingleStepOpt.getOpt());

                        if(line.hasOption(numIterationsOpt.getOpt())) {
                                maxIter = Integer.parseInt(line.getOptionValue(numIterationsOpt.getOpt()));
                        }

                        if(line.hasOption(numIterationsPerDocOpt.getOpt())) {
                                maxItersPerDoc = Integer.parseInt(line.getOptionValue(numIterationsPerDocOpt.getOpt()));
                        }

                        if(line.hasOption(blockSizeOpt.getOpt())) {
                                iterationBlockSize = Integer.parseInt(line.getOptionValue(blockSizeOpt.getOpt()));
                        }

                        if(line.hasOption(trainThreadsOpt.getOpt())) {
                                numTrainThreads = Integer.parseInt(line.getOptionValue(trainThreadsOpt.getOpt()));
                        }

                        if(line.hasOption(updateThreadsOpt.getOpt())) {
                                numUpdateThreads = Integer.parseInt(line.getOptionValue(updateThreadsOpt.getOpt()));
                        }

                        backfillPerplexity=line.hasOption(backfillPerplexityOpt.getOpt());
                        printAll=line.hasOption(printAllOpt.getOpt());

                        if(line.hasOption(testFractionOpt.getOpt())) {
                                testFraction = Float.parseFloat(line.getOptionValue(testFractionOpt.getOpt()));
                        }

                        if(line.hasOption(convergenceDeltaOpt.getOpt())) {
                                convergenceDelta = Double.parseDouble(line.getOptionValue(convergenceDeltaOpt.getOpt()));
                        }

                        if(line.hasOption(ldaSplitSizeOpt.getOpt())) {
                                ldaSplitSize = line.getOptionValue(ldaSplitSizeOpt.getOpt());
                        }
                        if(line.hasOption(numReduceTasksOpt.getOpt())) {
                                numReduceTasks = Integer.parseInt(line.getOptionValue(numReduceTasksOpt.getOpt()));
                        }
                        useTFIDF=line.hasOption(useTFIDFOpt.getOpt());
                        if(line.hasOption(maxDFPercentOpt.getOpt())) {
                                maxDFPercent = Integer.parseInt(line.getOptionValue(maxDFPercentOpt.getOpt()));
                        }
                        if(line.hasOption(minDFOpt.getOpt())) {
                                minDF = Integer.parseInt(line.getOptionValue(minDFOpt.getOpt()));
                        }



                        Option[] opts = line.getOptions();
                        for(Option o: opts) {
                                System.out.println(o.getDescription()+": "+o.getValue());
                        }

                        System.out.println("Running LDA Job:\n" +
                                "1-file location:"+baseFileLocation+"\n" +
                                "2-start at step:"+startAtStep + (singleStep?" (single step)":"") + "\n" +
                                "3-number of topics:"+numTopics+"\n" +
                                "4-number of iterations:"+maxIter+"\n" +
                                "5-number of iterations per document:"+maxItersPerDoc+"\n" +
                                "5-block size:"+ iterationBlockSize +"\n" +
                                "6-number of train threads:"+numTrainThreads+"\n" +
                                "7-number of update threads:"+numUpdateThreads+"\n" +
                                "8-number of reduce tasks:"+numReduceTasks+"\n");

                        Path output = new Path(baseFileLocation, "/output");
                        Configuration conf = new Configuration();
                        HadoopUtil.delete(conf, output);
                        String[] ldaArgs = {"-DbaseFileLocation=" + baseFileLocation};
                        // String[] strings =
                        // {"-Dmapred.input.dir=VectorFile/tfidf-vectors/part-r-00000"};
                        ToolRunner.run(new LDAJob(), ldaArgs);
                        System.out.println("done");
                }
        }

        public int run(String[] arg0) throws Exception {

                //16/03/23 19:42:55 INFO Configuration.deprecation: mapred.input.dir is deprecated. Instead, use mapreduce.input.fileinputformat.inputdir
                //16/03/23 19:42:55 INFO Configuration.deprecation: mapred.compress.map.output is deprecated. Instead, use mapreduce.map.output.compress
                //16/03/23 19:42:55 INFO Configuration.deprecation: mapred.output.dir is deprecated. Instead, use mapreduce.output.fileoutputformat.outputdir

                Configuration conf = getConf();
                String baseFileLocation = conf.get("baseFileLocation");
                Path input = new Path(baseFileLocation);
                System.out.println(input.toString());
                String seqFileOutput = ldaOutputPath +"SeqFile";
                String vectorOutFile = ldaOutputPath +"VectorFile";
                String rowIDOutFile = ldaOutputPath +"RowIdOutput";
                String ldaOutputFile = ldaOutputPath +"topicModelOutput";
                String dictionaryFileName = vectorOutFile + "/dictionary.file-0";
                String tempLDAModelFile = ldaOutputPath +"modelTemp";
                String docTopicOutput = ldaOutputPath +"docTopicOutput";
                String topicTermVectorDumpPath = ldaOutputPath +"topicTermVectorDump";
                String docTopicVectorDumpPath = ldaOutputPath +"docTopicVectorDump";

                // String topicTermVectorDump = "topicTermVectorDump";

                log.info("Deleting all the previous files.");
                if (startAtStep == 1) {
                        HadoopUtil.delete(conf, new Path(seqFileOutput));
                }
                if (startAtStep <= 2) {
                        HadoopUtil.delete(conf, new Path(vectorOutFile));
                }
                if (startAtStep <= 3) {
                        HadoopUtil.delete(conf, new Path(rowIDOutFile));
                }
                if(startAtStep <= 4) {
                        HadoopUtil.delete(conf, new Path(ldaOutputFile));
                        HadoopUtil.delete(conf, new Path(docTopicOutput));
                        HadoopUtil.delete(conf, new Path(tempLDAModelFile));
                }
                HadoopUtil.delete(conf, new Path(topicTermVectorDumpPath));
                HadoopUtil.delete(conf, new Path(docTopicVectorDumpPath));

                if(startAtStep==1) {
                        log.info("Step1: convert the directory into seqFile.");
                        System.out.println(ANSI_RED + "starting dir to seq job" + ANSI_RESET);
                        String[] dirToSeqArgs = {"--input", input.toString(), "--output",
                                seqFileOutput};
                        ToolRunner.run(new SequenceFilesFromDirectory(), dirToSeqArgs);
                        System.out.println("finished dir to seq job");
                }

                if((startAtStep<=2 && !singleStep) || startAtStep==2 ) {
                        log.info("Step 2: converting the seq to vector.");
                        System.out.println(ANSI_RED + "starting seq To Vector job" + ANSI_RESET);
                        /*String[] seqToVectorArgs = { "--input", seqFileOutput, "--output",
                                vectorOutFile, "--maxDFPercent", "70", "--maxNGramSize", "2",
                                "--namedVector", "--analyzerName",
                                "org.apache.lucene.analysis.core.WhitespaceAnalyzer" };*/
                        /*String[] seqToVectorArgs = { "--input", seqFileOutput, "--output",
                                vectorOutFile, "--maxDFPercent", "85", "--minDF", "2", "--maxNGramSize", "1", "--weight", "TF",
                                "--namedVector", "--analyzerName",
                                "org.apache.lucene.analysis.core.WhitespaceAnalyzer" };*/
                        String[] seqToVectorArgs = {"--input", seqFileOutput, "--output",
                                vectorOutFile, "--maxDFPercent", ""+maxDFPercent, "--minDF", ""+minDF, "--maxNGramSize", "1", "--weight", useTFIDF?"TFIDF":"TF",
                                "--namedVector", "--analyzerName",
                                "org.apache.lucene.analysis.core.WhitespaceAnalyzer"};
                        ToolRunner.run(new SparseVectorsFromSequenceFiles(), seqToVectorArgs);
                        System.out.println("finished seq to vector job");
                }

                if((startAtStep<=3 && !singleStep) || startAtStep==3) {
                        log.info("Step3: convert SequenceFile<Text, VectorWritable> to  SequenceFile<IntWritable, VectorWritable>");
                        System.out.println(ANSI_RED + "starting rowID job" + ANSI_RESET);
                /*String[] rowIdArgs = {
                        "-Dmapred.input.dir=" + vectorOutFile
                                + "/tfidf-vectors/part-r-00000",
                        "-Dmapred.output.dir=" + rowIDOutFile };*/
                        String[] rowIdArgs = {
                                "-Dmapred.input.dir=" + vectorOutFile
                                        + (useTFIDF?"/tfidf-vectors/part-r-00000":"/tf-vectors/part-r-00000"),
                                "-Dmapred.output.dir=" + rowIDOutFile};
                        ToolRunner.run(new RowIdJob(), rowIdArgs);
                        System.out.println("finished rowID job");
                }

                if((startAtStep<=4 && !singleStep) || startAtStep==4) {
                        log.info("Step4: Run the LDA algorithm");
                        System.out.println(ANSI_RED + "starting calculating the number of terms" + ANSI_RESET);

                        int numTerms = getNumTerms(conf, new Path(dictionaryFileName));
                        System.out.println("number of terms: " + numTerms);
                        long seed = System.nanoTime() % 10000;
                        System.out.println(ANSI_RED + "starting the CVB job" + ANSI_RESET);

                        //-Dmapred.max.split.size=10485760
                        conf.set("mapred.max.split.size",ldaSplitSize);

                        //CVB0Driver.run(conf, new Path("cvb/matrix"), mto, numTopics, numTerms, alpha, eta, maxIterations, iterationBlockSize, convergenceDelta, dictionaryPath, dto, msto, randomSeed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc, numReduceTasks, backfillPerplexity);
                        CVB0Driver cvbDriver = new CVB0Driver();
                        cvbDriver.run(conf, new Path(rowIDOutFile + "/matrix"), new Path(
                                        ldaOutputFile), numTopics, numTerms, docTopicSmoothening,
                                termTopicSmoothening, maxIter, iterationBlockSize,
                                convergenceDelta, new Path(dictionaryFileName), new Path(
                                        docTopicOutput), new Path(tempLDAModelFile), seed,
                                testFraction, numTrainThreads, numUpdateThreads,
                                maxItersPerDoc, numReduceTasks, backfillPerplexity);
                        //String[] runArgs ={};
                        System.out.println("finished the cvb job");
                }

                if(!singleStep || startAtStep==5) {

                        log.info("Step5: vectordump topic-term");

                        System.out.println(ANSI_RED + "starting the vector dumper for topic term" + ANSI_RESET);

                        VectorDumper.main(new String[]
                                {"-i",
                                        ldaOutputFile + "/part-m-00000",/* "-o",
                                ldaOutputFile + "/results",*/ "-d",
                                        dictionaryFileName, "-dt", "sequencefile",
                                        "-sort", "true", "-vs", "20"});


                        System.out.println("starting the vector dumper for topic term");
                        String[] topicTermDumperArg = {"--input", ldaOutputFile, /*"--output", topicTermVectorDumpPath,  */"--dictionary",
                                dictionaryFileName, "-dt", "sequencefile", "--vectorSize", "25", "-sort", "testsortVectors"};
                        ToolRunner.run(new Configuration(), new InternalVectorDumper(), topicTermDumperArg);
                        System.out.println("finished the vector dumper for topicterm");


                        if(printAll) {
                                System.out.println("starting the vector dumper for doctopic dumper");
                                String[] docTopicDumperArg = {"--input", docTopicOutput, /*"--output", docTopicVectorDumpPath*/};
                                ToolRunner.run(new Configuration(), new InternalVectorDumper(), docTopicDumperArg);
                        }

                }

                return 0;
        }

        private static int getNumTerms(Configuration conf, Path dictionaryPath) throws IOException {
                FileSystem fs = dictionaryPath.getFileSystem(conf);
                Text key = new Text();
                IntWritable value = new IntWritable();
                int maxTermId = -1;
                for (FileStatus stat : fs.globStatus(dictionaryPath)) {
                        SequenceFile.Reader reader = new SequenceFile.Reader(fs, stat.getPath(), conf);
                        while (reader.next(key, value)) {
                                maxTermId = Math.max(maxTermId, value.get());
                        }
                }
                return maxTermId + 1;
        }
}