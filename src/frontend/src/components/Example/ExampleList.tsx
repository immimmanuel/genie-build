import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "Start Trading",
    "learnig",
    "help", 
    "career", 
    "help", 
    "market Update", 
    "Companies and asset",
    "Emergency Center",
];

interface Props {
    onExampleClicked: (value: string) => void
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {DEFAULT_EXAMPLES.map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
