import React from 'react';
import styles from './LoadingScreen.module.css'; 

const LoadingScreen = ({ message = "Doing important stuff...", className = '' }) => {
  return (
    <div className={`${styles.loadingScreen} ${className}`}>
      <div className={styles.loadingWheel}></div>
      <p className={styles.loadingMessage}>{message}</p>
    </div>
  );
};

export default LoadingScreen;