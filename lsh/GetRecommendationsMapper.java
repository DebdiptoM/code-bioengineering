
package org.unigram.lsh;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.unigram.common.Candidate;
import org.unigram.common.RelatedUsersWritable;
import org.unigram.common.SeedClusterId;

/**
 * Mapper. 
 */
public class GetRecommendationsMapper extends
        Mapper<SeedClusterId, RelatedUsersWritable, LongWritable, Candidate> {
    
    /**
     * Map method.
     * 
     * @param key dummy
     * @param value related users
     * @param context for writing
     * @throws IOException -
     * @throws InterruptedException -
     */
    @Override
    public final void map(final SeedClusterId key,
            final RelatedUsersWritable value, final Context context) 
    throws IOException, InterruptedException  {
        List<LongWritable> relatedUsers = value.getRelatedUsers();
        for (int targetId = 0; targetId < relatedUsers.size(); targetId++) {
            this.writeCandidates(targetId, relatedUsers, context);
        }
    }

    /**
     * write candidates.
     * 
     * @param targetIndex target id
     * @param relatedUsers related users
     * @param context -
     * @throws IOException -
     * @throws InterruptedException -
     */
    private void writeCandidates(final int targetIndex,
            final List<LongWritable> relatedUsers, final Context context) 
        throws IOException, InterruptedException {
        LongWritable targetId 
            = new LongWritable(relatedUsers.get(targetIndex).get());        
        for (int candidateIndex = 0; 
            candidateIndex < relatedUsers.size(); candidateIndex++) {
            if (targetIndex == candidateIndex) {
                continue;
            }
            LongWritable candidateId 
                = new LongWritable(relatedUsers.get(candidateIndex).get());
            context.write(targetId, new Candidate(candidateId, 
                    new LongWritable(relatedUsers.size())));
        }
    }
}
