package si.david.mapreduce.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.mahout.clustering.lda.cvb.CVB0Driver;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.SequenceFilesFromDirectory;
import org.apache.mahout.utils.vectors.RowIdJob;
import org.apache.mahout.utils.vectors.VectorDumper;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by dstarina on 2/8/16.
 */
public class LDAJob extends AbstractJob {
        private static final Logger log = (Logger) LoggerFactory.getLogger(Job.class);
        static int numTopics = 20;
        static double doc_topic_smoothening = 0.0001;
        static double term_topic_smoothening = 0.0001;
        static int maxIter = 10;
        static int iteration_block_size = 10;
        static double convergenceDelta = 0;
        static float testFraction = 0.0f;
        static int numTrainThreads = 4;
        static int numUpdateThreads = 1;
        static int maxItersPerDoc = 10;
        static int numReduceTasks = 10;
        static boolean backfillPerplexity = false;



        public static void main(String args[]) throws Exception {
                String baseFileLocation = args[0];
                Path output = new Path(baseFileLocation,"/output");
                Configuration conf = new Configuration();
                HadoopUtil.delete(conf, output);
                //run(conf, new Path("/home/pacman/DownloadedCodes/mahout/output/ldaTest/inputFile"), output, new Path("resource/luceneDictionary") );

                run(conf, new Path(baseFileLocation, "/inputFile"), baseFileLocation);

        }

        private static void run(Configuration conf, Path input, String baseFileLocation) throws Exception {

                System.out.println(input.toString());
                String seqFileOutput = "SeqFile";
                String vectorOutFile = "VectorFile";
                String rowIDOutFile = "RowIdOutput";
                String ldaOutputFile = "topicModelOutputPath";
                String dictionaryFileName = vectorOutFile +"/dictionary.file-0";
                String tempLDAModelFile = "modelTempPath";
                String docTopicOutput = "docTopicOutputPath";
                String topicTermVectorDumpPath = "topicTermVectorDump";
                String docTopicVectorDumpPath = "docTopicVectorDump";

//      String topicTermVectorDump = "topicTermVectorDump";

                //Deleting all the previous files.
                HadoopUtil.delete(conf, new Path(seqFileOutput));
                HadoopUtil.delete(conf, new Path(vectorOutFile));
                HadoopUtil.delete(conf, new Path(rowIDOutFile));
                HadoopUtil.delete(conf, new Path(ldaOutputFile));
                HadoopUtil.delete(conf, new Path(docTopicOutput));
                HadoopUtil.delete(conf, new Path(tempLDAModelFile));
                HadoopUtil.delete(conf, new Path(topicTermVectorDumpPath));
                HadoopUtil.delete(conf, new Path(docTopicVectorDumpPath));



//      S3FileSystem.
                //Step1: convert the directory into seqFile.
                System.out.println("starting dir to seq job");
                String[] dirToSeqArgs = { "--input", input.toString(), "--output", seqFileOutput};
                ToolRunner.run(new SequenceFilesFromDirectory(), dirToSeqArgs);
                System.out.println("finished dir to seq job");

                //Step 2: converting the seq to vector.
                //TODO use the custome analyzer
                System.out.println("starting seq To Vector job");
                String[] seqToVectorArgs = {"--input", seqFileOutput, "--output",  vectorOutFile, "--maxDFPercent", "70", "--maxNGramSize", "2", "--namedVector", "--analyzerName", "com.ABC.mahout.clustering.CustomLuceneAnalyzer" };
                ToolRunner.run(new SparseVectorsFromSequenceFiles(), seqToVectorArgs);
                System.out.println("finished seq to vector job");

                //Step3: convert SequenceFile<Text, >> VectorWritable> to  SequenceFile<IntWritable, VectorWritable>
                System.out.println("starting rowID job");
                String[] rowIDArgs = {"--input",vectorOutFile + "/tfidf-vectors/part-r-00000", "--output", rowIDOutFile};
                ToolRunner.run(new RowIdJob(), rowIDArgs);
                System.out.println("finished rowID job");

                //Step4: Run the LDA algo
                //TODO use the termlist used in the LDAPrintTopWords.
                System.out.println("starting caluclulating the number of terms");
                int numTerms = getNumTerms(conf, new  Path(dictionaryFileName));
                System.out.println("finished calculating the number of terms");

                long seed = System.nanoTime() % 10000;
                System.out.println("starting the CVB job");

                CVB0Driver cvbDriver = new CVB0Driver();
                cvbDriver.run(conf, new Path(rowIDOutFile + "/matrix"),
                        new Path(ldaOutputFile), numTopics, numTerms, doc_topic_smoothening, term_topic_smoothening,
                        maxIter, iteration_block_size, convergenceDelta,
                        new Path(dictionaryFileName), new Path(docTopicOutput), new Path(tempLDAModelFile),
                        seed, testFraction, numTrainThreads, numUpdateThreads, maxItersPerDoc,
                        numReduceTasks, backfillPerplexity);
                System.out.println("finished the cvb job");
//

                //Step5: vectordump topic-term

                System.out.println("starting the vector dumper for topic term");
                String[] topicTermDumperArg = {"--input", ldaOutputFile, "--output", topicTermVectorDumpPath,  "--dictionary",
                        dictionaryFileName, "-dt", "sequencefile" ,"--vectorSize", "25" ,"-sort", "testsortVectors" };
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), topicTermDumperArg);
                System.out.println("finisher the vector dumper for topicterm");


                System.out.println("starting the vector dumper for doctopic dumper");
                String[] docTopicDumperArg = {"--input", docTopicOutput, "--output", docTopicVectorDumpPath};
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), docTopicDumperArg);
                System.out.println("finsiher the vector dumper for doctopic dumper");

                //
                //printLdaResults(ldaOutputFile, numTerms);
                //MongoDumper dumper = new MongoDumper();
                //dumper.writeTopicCollection(topicTermVectorDumpPath.toString());
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

        @Override
        public int run(String[] arg0) throws Exception {
                // TODO Auto-generated method stub
                return 0;
        }

}
