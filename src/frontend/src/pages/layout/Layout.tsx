import { Outlet, Link } from "react-router-dom";

import styles from "./Layout.module.css";


const Layout = () => {
    return (
        <div className={styles.layout}>
            <header className={styles.header} role={"banner"}>
                <div className={styles.headerContainer}>
                    <></>
                </div>
            </header>

            <Outlet />
        </div>
    );
};

export default Layout;
