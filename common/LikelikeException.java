
 
package org.unigram.common;


/**
 * LikelikeException.
 */
public class LikelikeException extends Exception {

    /**
     * Default constructor.
     */
    public LikelikeException() {
    }

    /**
     * Constructor.
     * @param cause the exception information to be added
     */
    public LikelikeException(final Throwable cause) {
        super(cause);
    }

    /**
     * Constructor.
     * @param details detailed information of exception
     * @param cause the exception information to be added
     */
    public LikelikeException(final String details,
            final Throwable cause) {
        super(details, cause);
    }

    /**
     * Constructor.
     * @param details detailed information of exception 
     */
    public LikelikeException(final String details) {
        super(details);
    }
    
}
