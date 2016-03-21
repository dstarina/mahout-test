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

public class LDAOutputJob extends AbstractJob {
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
        static int numTopics = 20;
        static double doc_topic_smoothening = 0.0001;
        static double term_topic_smoothening = 0.0001;
        // da pohitrim testiranje
        static int maxIter = 3;
        //static int maxIter = 10;
        static int iteration_block_size = 10;
        static double convergenceDelta = 0;
        static float testFraction = 0.0f;
        static int numTrainThreads = 4;
        static int numUpdateThreads = 1;
        //static int maxItersPerDoc = 10;
        // da pohitrim testiranje
        static int maxItersPerDoc = 3;
        static int numReduceTasks = 10;
        static boolean backfillPerplexity = false;

        public static void main(String args[]) throws Exception {
                // String baseFileLocation = args[0];
                String baseFileLocation = args[0];
                Path output = new Path(baseFileLocation, "/output");
                Configuration conf = new Configuration();
                HadoopUtil.delete(conf, output);
                String[] ldaArgs = { "-DbaseFileLocation=" + baseFileLocation };
                // String[] strings =
                // {"-Dmapred.input.dir=VectorFile/tfidf-vectors/part-r-00000"};
                ToolRunner.run(new LDAOutputJob(), ldaArgs);
                System.out.println("done");
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
                HadoopUtil.delete(conf, new Path(topicTermVectorDumpPath));
                HadoopUtil.delete(conf, new Path(docTopicVectorDumpPath));


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



                System.out.println(ANSI_RED+"1-starting the vector dumper for topic term"+ANSI_RESET);
                String[] topicTermDumperArg = {"--input", ldaOutputFile, /*"--output", topicTermVectorDumpPath,  */"--dictionary",
                        dictionaryFileName, "-dt", "sequencefile" ,"--vectorSize", "25" ,"-sort", "testsortVectors" };
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), topicTermDumperArg);
                System.out.println(ANSI_RED+"2-finished the vector dumper for topicterm"+ANSI_RESET);


                System.out.println(ANSI_RED+"3-starting the vector dumper for doctopic dumper"+ANSI_RESET);
                String[] docTopicDumperArg = {"--input", docTopicOutput, /*"--output", docTopicVectorDumpPath*/};
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), docTopicDumperArg);
                System.out.println(ANSI_RED+"4-finished the vector dumper for doctopic dumper"+ANSI_RESET);


                //VectorDumper.main(topicTermDumperArg); // David: zakomentiral
                System.out.println(ANSI_RED+"5-finished the vector dumper for topicterm"+ANSI_RESET);
                //System.out.println("starting the vector dumper for doctopic dumper");
                //String[] docTopicDumperArg = {"--input", docTopicOutput, "--output", docTopicVectorDumpPath};
                //ToolRunner.run(new Configuration(), new CustomVectorDumper(), docTopicDumperArg);
                //VectorDumper.main(docTopicDumperArg);
                System.out.println(ANSI_RED+"6-finished the vector dumper for doctopic dumper"+ANSI_RESET);

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