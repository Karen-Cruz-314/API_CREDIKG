import axios from 'axios';

function App() {
  const fetchItem = async () => {
    const response = await axios.get('http://localhost:8000/items/1');
    console.log(response.data);
  };

  return (
    <div>
      <button onClick={fetchItem}>Fetch Item</button>
    </div>
  );
}

export default App;