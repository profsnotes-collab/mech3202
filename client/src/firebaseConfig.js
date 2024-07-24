// src/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyAejR9MRXq9fcZm66RC6arxmqCmfOQD1BM",
  authDomain: "profsnotes-mech3202.firebaseapp.com",
  projectId: "profsnotes-mech3202",
  storageBucket: "profsnotes-mech3202.appspot.com",
  messagingSenderId: "495351986449",
  appId: "1:495351986449:web:f1d3e62c8c359f414334b0"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getFirestore(app);
const auth = getAuth(app);

export { db, auth };
