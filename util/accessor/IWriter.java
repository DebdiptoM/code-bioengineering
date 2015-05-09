
package org.unigram.util.accessor;

import org.apache.hadoop.mapreduce.Reducer.Context;

/**
 *
 */
public interface IWriter {
   @SuppressWarnings("unchecked")
   boolean write(Long key, Long value, Context context) throws Exception, InterruptedException;
}
