import { helper } from './legacy.cjs';

export function App() {
  return <button onClick={() => helper()}>go</button>;
}

export const Panel = () => <div onClick={() => helper()} />;
