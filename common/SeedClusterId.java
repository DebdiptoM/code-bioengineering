
package org.likelike.common;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

/**
 *
 */
public final class SeedClusterId 
    implements WritableComparable<SeedClusterId> {

    /**
     * Constructor.
     */
    public SeedClusterId() {}

    /**
     * Constructor.
     * @param seed hash seed
     * @param cid cluster id
     */
    public SeedClusterId(final long seed, 
            final long cid) {
        this.hashSeed = seed;
        this.clusterId = cid;
    }

    /**
     * get cluster id.
     * @return clusterId
     */
    public long getClusterId() {
        return clusterId;
    }
  
    /**
     * get hash seed.
     * @return hash seed
     */
    public long getSeed() {
        return hashSeed;
    }
  
    /**
     * write.
     * @param out output stream
     * @throws IOException -
     */
    @Override
    public void write(final DataOutput out) 
    throws IOException {
        out.writeLong(hashSeed);
        out.writeLong(clusterId);
    }
    
    /**
     * readFields.
     * 
     * @param in input stream
     * @throws IOException - 
     */
    @Override
    public void readFields(final DataInput in) 
        throws IOException {
        hashSeed = in.readLong();
        clusterId= in.readLong();
    }

    /**
     * Compare this into given one.
     * 
     * @param that another SeedClusterId object
     * @return -
     */
    @Override
    public int compareTo(final SeedClusterId that) {
        int seedResult = compareLongs(this.hashSeed, 
                that.getSeed());
        return seedResult == 0 ? compareLongs(clusterId, 
                that.getClusterId()) : seedResult;
    }

    /**
     * hasCode.
     * @return hashCode
     */
    @Override
    public int hashCode() {
        return (int) ((int) hashSeed + 53 * clusterId);
    }

    /**
     * equals.
     * @param o compared object
     * @return true when given object is the same as this. 
     * otherwise return false
     */
    @Override
    public boolean equals(final Object o) {
        if (o instanceof SeedClusterId) {
            SeedClusterId that = (SeedClusterId) o;
            return (hashSeed == that.getSeed()) 
            && (clusterId == that.getClusterId());
        }
        return false;
    }
    
    /**
     * toString.
     * @return string
     */
    @Override
    public String toString() {
        return hashSeed + ":" + clusterId;
    }
 
    /**
     * compare two long values.
     * @param a long
     * @param b long
     * @return 1 when a < b  
     */
    private static int compareLongs(final long a, final long b) {
        return a < b ? -1 : a > b ? 1 : 0;
    }
  
    /** hash seed value which is used to generate clusterId. */
    private long hashSeed;
  
    /** cluster ID. */
    private long clusterId; 
}