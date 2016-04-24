package si.david.mapreduce.lda;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by dstarina on 2/23/16.
 */
public class LDASimilarityJobBCK extends AbstractJob {

        public static final String ANSI_RESET = "\u001B[0m";
        public static final String ANSI_RED = "\u001B[31m";


        private static final Logger log = LoggerFactory.getLogger(Job.class);
        static int numTopics = 20;


        public static void main(String args[]) throws Exception {
                String[] similarityArgs = { "-DbaseFileLocation=" + args[0] };
                ToolRunner.run(new LDASimilarityJobBCK(), similarityArgs);
                System.out.println("done");
        }

        public int run(String[] arg0) throws Exception {

                /*
                //////// dimenzija
                Path inputMatrixPath = new Path(getInputPath());

                SequenceFile.Reader  sequenceFileReader =  new SequenceFile.Reader (fs, inputMatrixPath, conf);

                int NumberOfColumns = getDimensions(sequenceFileReader);

                sequenceFileReader.close();
                /////// end dimenzija
                */


                // čeprav: dimenzija je število tem, ki pa je bilo določeno ...

                /*
                hadoop jar mahout-examples-0.9-SNAPSHOT.jar
org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob
-i /user/dmilchev/wikipedia-matrix/matrix
-o /user/dmilchev/wikipedia-similarity
-r 4587604 --similarityClassname SIMILARITY_COSINE -m 50 -ess

                 */




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
                String rowSimilarityOutput = "rowSimilarity";


                log.info("Deleting all the previous files.");
                HadoopUtil.delete(conf, new Path(rowSimilarityOutput));


                System.out.println(ANSI_RED+"Starting to calculate similarity"+ANSI_RESET);


                //Command line arguments: {--endPhase=[2147483647], --excludeSelfSimilarity=[false]
                String[] rowSimilarityArgs = {"-i",  docTopicOutput, "-o", rowSimilarityOutput, "-r", ""+ numTopics,
                        "--similarityClassname", "SIMILARITY_EUCLIDEAN_DISTANCE", "-m", "10", "-ess"};

                ToolRunner.run(new RowSimilarityJob(), rowSimilarityArgs);


                // output
                System.out.println(ANSI_RED+"Similarity output:"+ANSI_RESET);
                String[] docTopicDumperArg = {"--input", rowSimilarityOutput, /*"--output", docTopicVectorDumpPath*/};
                ToolRunner.run(new Configuration(), new InternalVectorDumper(), docTopicDumperArg);

                return 0;
        }

        /*

        private int getDimensions(SequenceFile.Reader reader) throws IOException, InstantiationException,
                IllegalAccessException {
                Class keyClass = reader.getKeyClass();
                Writable row = (Writable) keyClass.newInstance();
                if (! reader.getValueClass().equals(VectorWritable.class)) {
                        throw new IllegalArgumentException("Value type of sequencefile must be a VectorWritable");
                }
                VectorWritable vw = new VectorWritable();
                if (!reader.next(row, vw)) {
                        log.error("matrix must have at least one row");
                        throw new IllegalStateException();
                }
                Vector v = vw.get();
                return v.size();
        }
        */
}
