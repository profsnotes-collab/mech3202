// src/firebaseConfig.js
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDPd86tojrZy0vr5xvBjEwWYc9l-20YUe4",
  authDomain: "prof-notes-ai.firebaseapp.com",
  projectId: "prof-notes-ai",
  storageBucket: "prof-notes-ai.appspot.com",
  messagingSenderId: "230906490116",
  appId: "1:230906490116:web:5cba3a180ef814b2df7451",
  measurementId: "G-HSNGFC8X53"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };
