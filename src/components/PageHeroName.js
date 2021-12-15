import React from "react"
import * as styles from "../styles/page-hero-name.module.scss"
export default function PageHeroName({ children }) {
	return <section className={styles.hero}>{children}</section>
}
