import React from 'react';
import styles from './LoadingScreen.module.css'; // We'll create this CSS module next

const LoadingScreen = ({ message = "Doing important stuff..." }) => {
  return (
    <div className={styles.loadingScreen}>
      <div className={styles.loadingWheel}></div>
      <p className={styles.loadingMessage}>{message}</p>
    </div>
  );
};

export default LoadingScreen;