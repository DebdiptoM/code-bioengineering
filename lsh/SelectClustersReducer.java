
package org.unigram.lsh;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;

import org.unigram.common.LikelikeConstants;
import org.unigram.common.RelatedUsersWritable;
import org.unigram.common.SeedClusterId;

/**
 * SelectClustersReducer. 
 */
public class SelectClustersReducer extends
        Reducer<SeedClusterId, RelatedUsersWritable,
        SeedClusterId, RelatedUsersWritable> {
    
    /**
     * Reduce.
     *
     * @param key cluster id
     * @param values user names
     * @param context -
     * @throws IOException -
     * @throws InterruptedException -
     */
    @Override
    public void reduce(final SeedClusterId key,
            final Iterable<RelatedUsersWritable> values,
            final Context context)
            throws IOException, InterruptedException {
       
        List<LongWritable> ids
            = new ArrayList<LongWritable>();

        for (RelatedUsersWritable relatedUsers : values) {
            List<LongWritable> tmpUsers = relatedUsers.getRelatedUsers();
            ids.addAll(tmpUsers);
            if (ids.size() >= this.maximumClusterSize) {
                break;
            }
        }
        if (this.minimumClusterSize <= ids.size()) {
            context.write(key, new RelatedUsersWritable(ids));
        }
    }
   
    /**
     * setup.
     * @param context -
     */
    @Override
    public final void setup(final Context context) {
        Configuration jc = context.getConfiguration();
        if (context == null || jc == null) {
            jc = new Configuration();
        }
        this.maximumClusterSize = jc.getLong(
                LikelikeConstants.MAX_CLUSTER_SIZE ,
                LikelikeConstants.DEFAULT_MAX_CLUSTER_SIZE);
        this.minimumClusterSize = jc.getLong(
                LikelikeConstants.MIN_CLUSTER_SIZE ,
                LikelikeConstants.DEFAULT_MIN_CLUSTER_SIZE);                
    }
   
    /** maximum number of examples in a cluster. */
    private long maximumClusterSize;
   
    /** minimum number of examples in a cluster. */    
    private long minimumClusterSize;
}

