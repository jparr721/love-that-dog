import app from 'firebase/app'
import 'firebase/auth';
import 'frebase/firestore';

const config = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_AUT_DOMAIN,
  databaseURL: process.env.REACT_APP_DATABASE_URL,
  projectId: `${process.env.REACT_APP_PROJECT_ID}`,
  storageBucket: process.env.REACT_APP_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_MESSAGING_SENDER_ID,
};

export default class Firebase {
  constructor() {
    app.initializeApp(config);
    this.db = app.firestore();
  }
}
