package si.david.mapreduce.lda;

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
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
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

public class LDA2Job extends AbstractJob {
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
        //static double doc_topic_smoothening = 0.0001;
        //static double term_topic_smoothening = 0.0001;
        static double doc_topic_smoothening = 50/numTopics; // alpha
        static double term_topic_smoothening = 50/numTopics; // eta
        // da pohitrim testiranje
        static int maxIter = 3;
        //static int maxIter = 10;
        static int iteration_block_size = 10;
        static double convergenceDelta = 0;
        //static float testFraction = 0.0f;
        static float testFraction = 0.005f;
        static int numTrainThreads = 4;
        static int numUpdateThreads = 1;
        //static int maxItersPerDoc = 10;
        // da pohitrim testiranje
        static int maxItersPerDoc = 3;
        static int numReduceTasks = 10;
        static boolean backfillPerplexity = false;

        static int startAtStep=1;

        public static void main(String args[]) throws Exception {
                // String baseFileLocation = args[0];
                if(args.length == 0) {
                        System.out.println("Parameters:\n1-file location\n2-start at step:\n\t1-beginning (convert the directory into seqFile)\n"
                                +"\t2-generate TF (converting the seq to vector)\n\t3-convert TF for LDA input\n\t4-LDA (Run the LDA algo)\n\t5-output (vectordump topic-term)"
                                +"\n3-number of topics("+numTopics+")\n4-number of iterations("+maxIter+")\n5-number of iterations per document("+maxItersPerDoc+")\n"
                                +"5-block size("+iteration_block_size+")\n6-number of train threads("+numTrainThreads+")\n7-number of update threads("+numUpdateThreads+")\n");
                } else {
                        int argSize = args.length;
                        if(argSize > 1) {
                                startAtStep = Integer.parseInt(args[1]);
                        }
                        if(argSize > 2) {
                                numTopics = Integer.parseInt(args[2]);
                                doc_topic_smoothening = 50/numTopics;
                                term_topic_smoothening = 50/numTopics;
                        }
                        if(argSize > 3) {
                                maxIter = Integer.parseInt(args[3]);
                        }
                        if(argSize > 4) {
                                maxItersPerDoc = Integer.parseInt(args[4]);
                        }
                        if(argSize > 5) {
                                iteration_block_size = Integer.parseInt(args[5]);
                        }
                        if(argSize > 6) {
                                numTrainThreads = Integer.parseInt(args[6]);
                        }
                        if(argSize > 7) {
                                numUpdateThreads = Integer.parseInt(args[7]);
                        }

                        String baseFileLocation = args[0];

                        System.out.println("Running LDA Job:\n1-file location:"+baseFileLocation+"\n2-start at step:"+startAtStep
                                +"\n3-number of topics:"+numTopics+"\n4-number of iterations:"+maxIter+"\n5-number of iterations per document:"+maxItersPerDoc+"\n"
                                +"5-block size:"+iteration_block_size+"\n6-number of train threads:"+numTrainThreads+"\n7-number of update threads:"+numUpdateThreads+"\n");

                        Path output = new Path(baseFileLocation, "/output");
                        Configuration conf = new Configuration();
                        HadoopUtil.delete(conf, output);
                        String[] ldaArgs = {"-DbaseFileLocation=" + baseFileLocation};
                        // String[] strings =
                        // {"-Dmapred.input.dir=VectorFile/tfidf-vectors/part-r-00000"};
                        ToolRunner.run(new LDA2Job(), ldaArgs);
                        System.out.println("done");
                }
        }

        public int run(String[] arg0) throws Exception {
                Configuration conf = getConf();
                // String baseFileLocation = "/Users/pin/java";
                String baseFileLocation = conf.get("baseFileLocation");
                //Path input = new Path(baseFileLocation, "/reuters-out");
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

                // S3FileSystem.
                if(startAtStep==1) {
                        log.info("Step1: convert the directory into seqFile.");
                        System.out.println(ANSI_RED + "starting dir to seq job" + ANSI_RESET);
                        String[] dirToSeqArgs = {"--input", input.toString(), "--output",
                                seqFileOutput};
                        ToolRunner.run(new SequenceFilesFromDirectory(), dirToSeqArgs);
                        System.out.println("finished dir to seq job");
                }

                if(startAtStep<=2) {
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

                if(startAtStep<=3) {
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

                if(startAtStep<=4) {
                        log.info("Step4: Run the LDA algo");
                        System.out.println(ANSI_RED + "starting caluclulating the number of terms" + ANSI_RESET);

                        int numTerms = getNumTerms(conf, new Path(dictionaryFileName));
                        System.out.println("finished calculating the number of terms");
                        long seed = System.nanoTime() % 10000;
                        System.out.println(ANSI_RED + "starting the CVB job" + ANSI_RESET);

                        //CVB0Driver.run(conf, new Path("cvb/matrix"), mto, numTopics, numTerms, alpha, eta, maxIterations, iterationBlockSize, convergenceDelta, dictionaryPath, dto, msto, randomSeed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc, numReduceTasks, backfillPerplexity);
                        CVB0Driver cvbDriver = new CVB0Driver();
                        cvbDriver.run(conf, new Path(rowIDOutFile + "/matrix"), new Path(
                                        ldaOutputFile), numTopics, numTerms, doc_topic_smoothening,
                                term_topic_smoothening, maxIter, iteration_block_size,
                                convergenceDelta, new Path(dictionaryFileName), new Path(
                                        docTopicOutput), new Path(tempLDAModelFile), seed,
                                testFraction, numTrainThreads, numUpdateThreads,
                                maxItersPerDoc, numReduceTasks, backfillPerplexity);
                        //String[] runArgs ={};
                        System.out.println("finished the cvb job");
                }

                log.info("Step5: vectordump topic-term");

                System.out.println(ANSI_RED+"starting the vector dumper for topic term"+ANSI_RESET);
                //String[] topicTermDumperArg = {"--seqFile", ldaOutputFile+"/part-m-00000",  "--dictionary",
                //        dictionaryFileName, "-dt", "sequencefile"  };



                //ToolRunner.run(new Configuration(), new CustomVectorDumper(), topicTermDumperArg);
                //VectorDumper.main(topicTermDumperArg);
                //SequenceFileDumper.main(topicTermDumperArg);
                //String[] topicTermDumperArg = {"--input", ldaOutputFile, "--output", topicTermVectorDumpPath,  "--dictionary",
                //        dictionaryFileName, "-dt", "sequencefile" ,"--vectorSize", "25" ,"-sort", "testsortVectors" };
                //LDAPrintTopics.main(topicTermDumperArg);
                //String[] topicTermDumperArg = {"-seq"};

/*
                for(int k=0;k<numTopics;k++){
                        System.out.println(ANSI_RED+"Dumping topic \t"+k+ANSI_RESET);
                        String partFile="part-m-0000"+k;
                        if(k>=10)
                                partFile="part-m-000"+k;

                        String output="topic"+k;
                        String[] topicTermDumperArg2 = {"-s", ldaOutputFile+"/"+partFile, "-dt", "sequencefile", "-d",
                                dictionaryFileName, "-o",output,  "-c", };

                        VectorDumper.main(topicTermDumperArg2);

                }*/



                VectorDumper.main(new String[]
                        { "-i",
                                ldaOutputFile + "/part-m-00000",/* "-o",
                                ldaOutputFile + "/results",*/ "-d",
                                dictionaryFileName, "-dt", "sequencefile",
                                "-sort", "true", "-vs", "20" });
/*
                VectorDumper.main(new String[]
                        { "-i",
                                ldaOutputFile + "/topic-term-dist/part-m-00000", "-o",
                                ldaOutputFile + "/results", "-d",
                                ldaOutputFile + "/dictionary.file-0", "-dt", "sequencefile",
                                "-sort", "true", "-vs", "20" });*/



                System.out.println("starting the vector dumper for topic term");
                String[] topicTermDumperArg = {"--input", ldaOutputFile, /*"--output", topicTermVectorDumpPath,  */"--dictionary",
                        dictionaryFileName, "-dt", "sequencefile" ,"--vectorSize", "25" ,"-sort", "testsortVectors" };
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), topicTermDumperArg);
                System.out.println("finisher the vector dumper for topicterm");


                System.out.println("starting the vector dumper for doctopic dumper");
                String[] docTopicDumperArg = {"--input", docTopicOutput, /*"--output", docTopicVectorDumpPath*/};
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), docTopicDumperArg);
                System.out.println("finsiher the vector dumper for doctopic dumper");


                //VectorDumper.main(topicTermDumperArg); // David: zakomentiral
                System.out.println("finished the vector dumper for topicterm");
                //System.out.println("starting the vector dumper for doctopic dumper");
                //String[] docTopicDumperArg = {"--input", docTopicOutput, "--output", docTopicVectorDumpPath};
                //ToolRunner.run(new Configuration(), new CustomVectorDumper(), docTopicDumperArg);
                //VectorDumper.main(docTopicDumperArg);
                System.out.println("finished the vector dumper for doctopic dumper");

                //printLdaResults(ldaOutputFile, numTerms);
                //MongoDumper dumper = new MongoDumper();
                //dumper.writeTopicCollection(topicTermVectorDumpPath.toString());
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