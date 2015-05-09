
package org.unigram.likelike.lsh.function;

import java.util.Set;

import org.apache.hadoop.io.LongWritable;

/**
 * Interface hash function for LSH. 
 */
public interface IHashFunction {
    /**
     * Compute hashing function.
     *  
     * @param featureVector feature vector
     * @param seed hash seed
     * @return hashed value
     */
    LongWritable returnClusterId(Set<Long> featureVector, 
            long seed);
}
