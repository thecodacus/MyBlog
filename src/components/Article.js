import React from "react"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import * as styles from "../styles/post-details.module.scss"
export default function Article({ html, title, category, featuredImage, date, publicURL, children }) {
	const featuredImagePath = featuredImage?.publicURL || publicURL
	return (
		<article className={styles.details}>
			<h1>{title}</h1>
			<div className={styles.metadata}>
				<h3>{new Date(date).toDateString()}</h3>
				<h3>.</h3>
				<h3>{category}</h3>
			</div>

			<div className={styles.featured}>
				{featuredImage?.childImageSharp ? (
					<GatsbyImage image={getImage(featuredImage)} alt={title} />
				) : (
					<img className={styles.alternateImage} src={featuredImagePath} alt={title} />
				)}
			</div>
			<div className={styles.html} dangerouslySetInnerHTML={{ __html: html }}></div>
			{children}
		</article>
	)
}
