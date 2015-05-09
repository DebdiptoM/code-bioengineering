
package org.unigram.likelike.common;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.logging.Level;

/**
 * File utility class. 
 */
public final class FsUtil {
    
    /** logger. */
    private static LikelikeLogger logger =LikelikeLogger.getLogger();

    /**
     * for safe.
     */
    private FsUtil() {
        //dummy
    }

   /**
    * Check a file or directory exist. if so delete them.
    * @param dir path to be checked
    * @param fs filesystem containing the path
    * @return true when the check succeeded
    * @throws IOException occurs deleting file
    */
   public static boolean checkPath(final Path dir,
           final FileSystem fs)
   throws IOException {
        if (fs.exists(dir)) {
            logger.log(Level.INFO, "Overriding: " + dir.toString());
            return fs.delete(dir, true);
        } else {
            logger.log(Level.FINE, "No such file: " + dir.toString());
            return true;
        }
    }
      
   /**
    * Check a file or directory exist. if so delete them.
    *
    * @param dir dir path to be checked
    * @param conf containing the filesystem of path
    * @return true when the check succeeded
    * @throws IOException when opening error such as there is no directory. 
    */
   public static boolean checkPath(final Path dir, 
           final Configuration conf)
   throws IOException {
       return checkPath(dir, FileSystem.get(conf));
    }
   
   /**
    * Delete files.  
    * 
    * @param fs filesytem containing files with fileNames
    * @param fileNames file names to be removed
    * @throws IOException -
    */
   public static void clean(final FileSystem fs, 
       final String... fileNames) throws IOException {

       for (String fileName : fileNames) {
           Path path = new Path(fileName);
           if (fs.exists(path)) {
               logger.log(Level.INFO,
                       "Removing: " + path.toString());
               fs.delete(path, true);
           }
       }
   }

}
