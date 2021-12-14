import React from "react"
import * as styles from "../styles/posts.module.scss"
import { GatsbyImage, getImage } from "gatsby-plugin-image"
import { Link } from "gatsby"
export default function PostsGrid({ posts }) {
	return (
		<div className={styles.posts}>
			{posts.map(post => (
				<Link to={"/posts/" + post.fields.slug} key={post.id}>
					<div className={styles.card}>
						<div className={styles.thumbnail}>
							{post.frontmatter.featuredImage.childImageSharp != null ? (
								<GatsbyImage image={getImage(post.frontmatter.featuredImage)} alt={post.frontmatter.title} />
							) : (
								<img className={styles.alternateImage} src={post.frontmatter.featuredImage.publicURL} alt={post.frontmatter.title} />
							)}
						</div>
						<span className={styles.chip}>{post.frontmatter.category}</span>
						<div className={styles.excerpt}>
							<h3>{post.frontmatter.title}</h3>
							<p>{post.excerpt}</p>
						</div>
						<div className={styles.footer}>
							<div style={{ flex: 1 }}>
								<p>Date: {new Date(post.frontmatter.date).toLocaleDateString()}</p>
							</div>
							<div className={styles.readMore}>Read More</div>
						</div>
					</div>
				</Link>
			))}
		</div>
	)
}
