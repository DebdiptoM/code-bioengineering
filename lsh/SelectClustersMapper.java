
package org.unigram.lsh;
         
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.unigram.common.LikelikeConstants;
import org.unigram.common.RelatedUsersWritable;
import org.unigram.common.SeedClusterId;
import org.unigram.lsh.function.IHashFunction;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashSet;
import java.util.Set;

/**
 * SelectClustersMapper.
 */
public class SelectClustersMapper extends
        Mapper<LongWritable, Text, SeedClusterId, RelatedUsersWritable> {
    /**
     * map.
     * @param key dummy
     * @param value containing id and the features
     * @param context context 
     * @exception IOException -
     * @exception InterruptedException -
     */
    @Override
    public final void map(final LongWritable key,
            final Text value, final Context context)
            throws IOException, InterruptedException {
        String inputStr = value.toString();

        try {
            String[] tokens = inputStr.split("\t");
            Long id = Long.parseLong(tokens[0]); // example id
            Set<Long> featureSet 
                = this.extractFeatures(tokens[1]);
            
            for (int i=0; i<seedsAry.length; i++) {
                LongWritable clusterId 
                    = this.function.returnClusterId(featureSet, 
                        seedsAry[i]);
                context.write(new SeedClusterId(
                        seedsAry[i], clusterId.get()), 
                        new RelatedUsersWritable(id)); 
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("PARSING ERROR in line: " + inputStr);
            e.printStackTrace();
        }
    }
    
    /**
     * Extract features from feature string.
     * 
     * @param featureStr string containing feature information
     * @return map containing feature and the value
     */
    private Set<Long> extractFeatures(
            final String featureStr) {
        Set<Long> rtSet = new HashSet<Long>();
        String[] featureArray = featureStr.split(" ");
        for (int i=0; i<featureArray.length; i++) {
            rtSet.add(Long.parseLong(featureArray[i]));
        }
        return rtSet;
    }
    
    /**
     * setup.
     * @param context context
     */
    public final void setup(final Context context) {
        Configuration jc = context.getConfiguration();

        /* create a object implements IHashFunction */
        String functionClassName 
            = LikelikeConstants.DEFAULT_HASH_FUNCTION;
        if (context == null || jc == null) {
            /* added default configuration for testing */
            jc = new Configuration();
        }
        try {
            functionClassName = jc.get(
                    LikelikeConstants.HASH_FUNCTION,
                    LikelikeConstants.DEFAULT_HASH_FUNCTION);
            Class<? extends IHashFunction> functionClass 
                = Class.forName(
                    functionClassName).asSubclass(
                            IHashFunction.class);
            Constructor<? extends IHashFunction> constructor 
                = functionClass
                    .getConstructor(Configuration.class);
            function = constructor.newInstance(jc);
        } catch (NoSuchMethodException | ClassNotFoundException |
                InstantiationException | IllegalAccessException |
                InvocationTargetException e){
            throw new RuntimeException(e);
        }
        
        /* extract set of hash seeds */
        String seedsStr = jc.get(MINWISE_HASH_SEEDS, 
                DEFAULT_MINWISE_HASH_SEEDS);
        String[] seedsStrAry = seedsStr.split(":");
        this.seedsAry = new long[seedsStrAry.length];
        for (int i =0; i< seedsStrAry.length; i++) {
            this.seedsAry[i] =Long.parseLong(seedsStrAry[i]);
        }
    }
        
    /** Hash function object. */
    private IHashFunction function;
    
    /** Set of hash seeds. */
    private long[] seedsAry; 

    /** Symbol: hash seed. */
    public static final String MINWISE_HASH_SEEDS
        = "likelike.minwise.hash.seedS";
    
    /** Default: hash seed. */
    public static final String DEFAULT_MINWISE_HASH_SEEDS    
        = "1";        
    
}
