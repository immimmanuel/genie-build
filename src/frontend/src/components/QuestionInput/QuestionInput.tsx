import { useEffect, useState } from "react";
import { Stack, TextField } from "@fluentui/react";
import { Button, Tooltip } from "@fluentui/react-components";
import { Send28Filled, Attach28Filled } from "@fluentui/react-icons";

import styles from "./QuestionInput.module.css";

interface Props {
    onSend: (question: string) => void;
    disabled: boolean;
    initQuestion?: string;
    placeholder?: string;
    clearOnSend?: boolean;
}

export const QuestionInput = ({ onSend, disabled, placeholder, clearOnSend, initQuestion }: Props) => {
    const [question, setQuestion] = useState<string>("");

    useEffect(() => {
        initQuestion && setQuestion(initQuestion);
    }, [initQuestion]);

    const sendQuestion = () => {
        if (disabled || !question.trim()) {
            return;
        }

        onSend(question);

        if (clearOnSend) {
            setQuestion("");
        }
    };

    const onEnterPress = (ev: React.KeyboardEvent<Element>) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
            ev.preventDefault();
            sendQuestion();
        }
    };

    const addAttachment = () => {
        console.log("Add attachment");
    }

    const onQuestionChange = (_ev: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        if (!newValue) {
            setQuestion("");
        } else if (newValue.length <= 1000) {
            setQuestion(newValue);
        }
    };

    return (
        <Stack horizontal className={styles.questionInputContainer}>
            <Button size="large" icon={<Attach28Filled primaryFill="rgb(137, 139, 173)" />} onClick={sendQuestion} />
            <TextField
                className={styles.questionInputTextArea}
                placeholder={placeholder}
                multiline
                resizable={false}
                borderless
                value={question}
                onChange={onQuestionChange}
                onKeyDown={onEnterPress}
            />
            <div className={styles.questionInputButtonsContainer}>
                <Tooltip content="Ask question button" relationship="label">
                    <Button size="large" icon={<Send28Filled primaryFill="rgb(228, 229, 238)" />} onClick={addAttachment} />
                </Tooltip>
            </div>
        </Stack>
    );
};
