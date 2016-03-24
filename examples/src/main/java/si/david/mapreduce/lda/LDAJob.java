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

import java.io.File;
import java.io.IOException;

/**
 * Created by dstarina on 2/8/16.
 */

public class LDAJob extends AbstractJob {
        public static final String ANSI_RESET = "\u001B[0m";
        public static final String ANSI_BLACK = "\u001B[30m";
        public static final String ANSI_RED = "\u001B[31m";
        public static final String ANSI_GREEN = "\u001B[32m";
        public static final String ANSI_YELLOW = "\u001B[33m";
        public static final String ANSI_BLUE = "\u001B[34m";
        public static final String ANSI_PURPLE = "\u001B[35m";
        public static final String ANSI_CYAN = "\u001B[36m";
        public static final String ANSI_WHITE = "\u001B[37m";


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
                Option backfillPerplexityOpt = new Option("bf", "backfillPerplexity", false, "backfill perplexity");
                Option printAllOpt = new Option("pa","print all results");

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
                                String fileName="mahout-examples-0.11.1-job.jar";
                                try {
                                        fileName = new File(LDAJob.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath()).getName();
                                } catch (Exception e) { }
                                formatter.printHelp( "Location is required. Usage: hadoop jar "+fileName+" si.david.mapreduce.lda.LDAJob", options );

                                return;
                        }

                        if(line.hasOption("s")) {
                                startAtStep = Integer.parseInt(line.getOptionValue("s"));
                        }

                        if(line.hasOption("t")) {
                                numTopics = Integer.parseInt(line.getOptionValue("t"));
                                docTopicSmoothening = 50/numTopics;
                                termTopicSmoothening = 50/numTopics;
                        }

                        singleStep=line.hasOption("st");

                        if(line.hasOption("i")) {
                                maxIter = Integer.parseInt(line.getOptionValue("i"));
                        }

                        if(line.hasOption("d")) {
                                maxItersPerDoc = Integer.parseInt(line.getOptionValue("d"));
                        }

                        if(line.hasOption("b")) {
                                iterationBlockSize = Integer.parseInt(line.getOptionValue("b"));
                        }

                        if(line.hasOption("tr")) {
                                numTrainThreads = Integer.parseInt(line.getOptionValue("tr"));
                        }

                        if(line.hasOption("u")) {
                                numUpdateThreads = Integer.parseInt(line.getOptionValue("u"));
                        }

                        backfillPerplexity=line.hasOption("bf");
                        printAll=line.hasOption("pa");

                        if(line.hasOption("tf")) {
                                testFraction = Float.parseFloat(line.getOptionValue("tf"));
                        }

                        if(line.hasOption("cd")) {
                                convergenceDelta = Double.parseDouble(line.getOptionValue("cd"));
                        }

                        if(line.hasOption("ls")) {
                                ldaSplitSize = line.getOptionValue("ls");
                        }


                        System.out.println("Running LDA Job:\n" +
                                "1-file location:"+baseFileLocation+"\n" +
                                "2-start at step:"+startAtStep + (singleStep?" (single step)":"") + "\n" +
                                "3-number of topics:"+numTopics+"\n" +
                                "4-number of iterations:"+maxIter+"\n" +
                                "5-number of iterations per document:"+maxItersPerDoc+"\n" +
                                "5-block size:"+ iterationBlockSize +"\n" +
                                "6-number of train threads:"+numTrainThreads+"\n" +
                                "7-number of update threads:"+numUpdateThreads+"\n");

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
                String seqFileOutput = "SeqFile";
                String vectorOutFile = "VectorFile";
                String rowIDOutFile = "RowIdOutput";
                String ldaOutputFile = "topicModelOutputPath";
                String dictionaryFileName = vectorOutFile + "/dictionary.file-0";
                String tempLDAModelFile = "modelTempPath";
                String docTopicOutput = "docTopicOutputPath";
                String topicTermVectorDumpPath = "topicTermVectorDump";
                String docTopicVectorDumpPath = "docTopicVectorDump";

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
                                vectorOutFile, "--maxDFPercent", "85", "--minDF", "2", "--maxNGramSize", "1", "--weight", "TF",
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
                                        + "/tf-vectors/part-r-00000",
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